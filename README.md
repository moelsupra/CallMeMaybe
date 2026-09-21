*This project has been created as part of the 42 curriculum by moelamma.*

# CallMeMaybe - Function Calling with Constrained Decoding

CallMeMaybe is a lightweight, reliable function-calling engine that translates natural language requests into typed, structured function calls using **constrained decoding** on open-weight language models.

Unlike standard prompt-engineering approaches that rely on the LLM to write raw JSON, CallMeMaybe enforces syntactic and structural constraints directly at the model's **logit generation layer**. This guarantees 0% hallucinated function names and produces 100% valid, parseable JSON outputs.

---

## Description

Modern LLM-based agent systems often require models to trigger external tools (APIs, mathematical functions, database queries). Standard text generation prompting faces three critical reliability issues:
1. **Hallucination**: The model invents non-existent function names or invalid parameter keys.
2. **Syntax Errors**: Generated JSON output may contain unescaped quotes, trailing commas, or missing brackets.
3. **Type Inconsistencies**: Numeric parameters may be returned as words (`"two"`), unparseable equations, or improperly formatted types.

**CallMeMaybe** solves these issues by intercepting the model at the vocabulary prediction layer:
* Only valid function tokens are allowed into the sampling space by applying a $-\infty$ mask over disallowed token logits.
* Parameters are decoded according to their schema type (numbers, integers, booleans, strings).
* Output data is assembled into strongly typed Pydantic models and serialized using standard Python JSON encoders.

---

## Instructions

### Prerequisites
- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) (fast Python package and project manager)

### Installation
Sync project dependencies using `uv`:
```bash
uv sync
```

### Execution
The project can be executed directly through `uv`:
```bash
uv run python -m src
```

Or using the provided `Makefile`:
```bash
make run
```

### Custom Arguments
You can override default file paths via command-line flags:
```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

| Flag | Default | Description |
|---|---|---|
| `--functions_definition` | `data/input/functions_definition.json` | Path to JSON schema containing callable functions |
| `--input` | `data/input/function_calling_tests.json` | Path to JSON containing input natural language prompts |
| `--output` | `data/output/function_calling_results.json` | Destination path for resulting JSON function calls |

### Development & Quality Checks
Run formatting, linting, and static type checking:
```bash
make lint        # Runs flake8 and mypy with type checks
make clean       # Removes build artifacts and cache folders
```

---

## Algorithm Explanation

The core engine in [`src/decoder.py`](src/decoder.py) implements token-level constrained decoding.

```
                    [ Input Prompt + Available Functions ]
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  select_function()            │
                       │  - Prefix Trie Matching       │
                       │  - Logit -inf Masking         │
                       │  - Deterministic Auto-Advance │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                              Selected Function
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  extract_parameters()         │
                       │  - Number: digit token mask   │
                       │  - Integer: no dot (.) mask   │
                       │  - Boolean: true vs false     │
                       │  - String: quote stop tokens  │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  Validated FunctionCallResult │
                       │  -> 100% Valid JSON Output    │
                       └───────────────────────────────┘
```

### 1. Function Selection via Prefix-Constrained Logit Masking
1. **Pre-tokenization**: Each candidate function name (e.g. `fn_add_numbers`) is encoded into a list of token IDs (`fn_tokens`).
2. **Prefix Filtering**: At step $t$, the generator inspects tokens generated so far (`generated_tokens`). It filters candidate functions whose prefix matches `generated_tokens` and extracts the next possible token for each candidate (`allowed_tokens`).
3. **Deterministic Fast-Path**: If `len(allowed_tokens) == 1`, only one branch is possible. The token is appended immediately without querying the LLM, reducing latency.
4. **Logit Masking**: If `len(allowed_tokens) > 1`:
   - Raw logits are retrieved from `model.get_logits_from_input_ids()`.
   - A mask vector initialized to $-\infty$ is created.
   - Only the indices corresponding to `allowed_tokens` keep their model logits.
   - `argmax(mask)` selects the highest probability token among valid candidates:
     $$\text{mask}_i = \begin{cases} \text{logits}_i & \text{if } i \in \text{allowed\_tokens} \\ -\infty & \text{otherwise} \end{cases}$$
5. **Termination**: When no further continuation tokens exist (`allowed_tokens` is empty), generation halts and matches the tokens to the exact function name.

### 2. Typed Parameter Extraction
- **`number` (Float)**: Restricts token generation strictly to digits, minus sign, decimal dot, and delimiters (`0123456789.-`, `,`, `}`). All non-numeric tokens receive $-\infty$ logits.
- **`integer`**: Reuses the numeric constrained loop, but restricts tokens to `0123456789-` (excluding `.`), guaranteeing integer-only representation before casting with `int()`.
- **`boolean`**: Directly queries the model logits for `"true"` vs `"false"` (`logits[true_tid] > logits[false_tid]`), choosing the higher probability state in a single step.
- **`string`**: Greedily generates tokens until encountering a quote delimiter (`"`) or newline, cleanly extracting string arguments.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Logit-level Masking instead of Free-Text Prompting** | Eliminates hallucination entirely. The model cannot output an invalid function name or ill-typed token even under adversarial prompts. |
| **Trie Auto-Advance Optimization** | When only one token continuation exists in the prefix tree, inference is bypassed. This saves inference passes and speeds up multi-token function selection. |
| **Strict Pydantic Validation (`extra="forbid"`)** | Detects schema mismatches, malformed structures, or unrecognized keys in input configuration files before execution begins. |
| **Duplicate Key Detection (`object_pairs_hook`)** | Python's standard `json.loads` silently overwrites duplicate JSON keys. Using `raise_on_duplicate_keys` ensures strict JSON compliance. |
| **Python Standard Library JSON Serialization** | Instead of asking the LLM to write raw JSON syntax, values are extracted as native Python types and serialized using `json.dump()`. This guarantees 100% parseable, RFC 8259-compliant JSON. |

---

## Performance Analysis

- **Accuracy**: 100% accuracy on valid function name selection and typing constraints. By construction, the probability of selecting an illegal function name or invalid numeric character is $0$ ($e^{-\infty} = 0$).
- **Speed**: Optimized token decoding minimizes unnecessary forward passes. By skipping LLM evaluation when `len(allowed_tokens) == 1`, common prefixes (like `fn_` or `ft_`) advance instantly.
- **Reliability**: Graceful error handling in [`src/loader.py`](src/loader.py) catches missing files, empty inputs, duplicate identifiers, and schema errors, outputting descriptive messages to `stderr` with status code 1 instead of raw python tracebacks.

---

## Challenges Faced & Solutions

1. **Token Boundary Mismatches**:
   - *Problem*: Different tokenizers split words differently depending on context (e.g., `"add"` vs `" add"` vs `"_add"`).
   - *Solution*: Pre-encode complete function names standalone and build prefix trees on the exact token IDs produced by the model tokenizer.

2. **Numeric Token Discretization in Qwen3**:
   - *Problem*: Models like Qwen3 tokenize numbers character-by-character (`"42"` $\rightarrow$ `['4', '2']`).
   - *Solution*: Pre-computed token IDs for individual digits `0–9`, decimal point `.`, and negative sign `-`, allowing token-by-token number accumulation with stop tokens (`,`, `}`, space).

3. **Graceful Error Handling without Stack Dumps**:
   - *Problem*: Evaluator test suites often supply corrupt JSON files or invalid schemas to verify application stability.
   - *Solution*: Centralized `handle_error()` captures `FileNotFoundError`, `json.JSONDecodeError`, `pydantic.ValidationError`, and general exceptions, displaying user-friendly error diagnostics and clean exits.

---

## Testing Strategy

The project underwent rigorous testing across multiple dimensions:

1. **Static Analysis & Linting**:
   - `flake8`: Strict PEP 8 compliance, unused import removal, and clean line lengths.
   - `mypy`: Strict typing with `--disallow-untyped-defs`, `--check-untyped-defs`, and `--warn-return-any`.
2. **Input Schema Validation**:
   - Duplicate keys in function definition files.
   - Invalid Python identifiers in function or parameter names.
   - Empty input arrays and missing files.
3. **End-to-End Execution**:
   - Verified end-to-end translation on test suites in `data/input/function_calling_tests.json`.
   - Verified that the resulting JSON adheres to the required output schema:
     ```json
     [
       {
         "prompt": "What is the sum of 2 and 3?",
         "name": "fn_add_numbers",
         "parameters": {
           "a": 2.0,
           "b": 3.0
         }
       }
     ]
     ```

---

## Example Usage

### Running the Default Pipeline
```bash
uv run python -m src
```

### Console Output
```text
🤖 Starting CallMeMaybe...
Loaded 5 functions and 11 prompts.
  ✅ fn_add_numbers({'a': 2.0, 'b': 3.0})
  ✅ fn_add_numbers({'a': 265.0, 'b': 345.0})
  ✅ fn_greet({'name': 'shrek'})
  ✅ fn_reverse_string({'s': 'hello'})
  ✅ fn_get_square_root({'a': 16.0})

📄 Output written to data/output/function_calling_results.json
```

---

## Resources & AI Usage

### References, Tutorials & Documentation
- [Constrained Decoding Tutorial / Video Reference 1](https://youtu.be/LPZh9BOjkQs?si=M_aF_6M0_FooE_Zm)
- [Constrained Decoding Tutorial / Video Reference 2](https://youtu.be/Nh6qoBnreBc?si=HeOfJ7rny7gLfzgw)
- [Constrained Decoding Tutorial / Video Reference 3](https://youtu.be/xpvFinvqRCA?si=YT0pmBCHHEqGjIYA)
- [Hugging Face Transformers: LogitsProcessor & Generation](https://huggingface.co/docs/transformers/internal/generation_utils)
- [Outlines: Structured Text Generation](https://github.com/dottxt-ai/outlines)
- [Pydantic Documentation: Models & Validation](https://docs.pydantic.dev/latest/)
- [JSON Schema Specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259)

### AI Usage Disclosure
In accordance with 42 curriculum guidelines, AI assistance was used during this project for:
- **Architecture Consultation & Research**: Discussing constrained decoding algorithms (trie traversal vs. logit masking tradeoffs).
- **Code Refactoring**: Optimizing numeric and boolean parameter extraction in `src/decoder.py` to eliminate code duplication.
- **Documentation & Linting Review**: Structuring the README, refining technical explanations, and ensuring compliance with flake8 and mypy type checks.
