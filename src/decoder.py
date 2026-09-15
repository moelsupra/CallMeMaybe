"""Constrained decoding engine for CallMeMaybe.

Guarantees valid function selection and typed argument extraction
token-by-token using the model's logits and schema constraints.
"""

import numpy as np
from typing import Any
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.models import FunctionCallResult, FunctionDefinition


def select_function(
    model: Small_LLM_Model,
    prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Select the best matching function using constrained decoding.
    Args:
        model: Loaded Small_LLM_Model instance.
        prompt: The user natural language request.
        functions: List of available FunctionDefinition objects.
    Returns:
        The exact string name of the selected function.
    """

    fn_tokens: dict[str, list[int]] = {
        fn.name: model.encode(fn.name)[0].tolist() for fn in functions
    }

    fn_lines = "\n".join(
        [f"- {fn.name}: {fn.description}" for fn in functions]
    )

    system_prompt = (
        f"Available functions:\n{fn_lines}\n\n"
        f"User request: {prompt}\n"
        f"Function:"
    )

    input_ids = model.encode(system_prompt)[0].tolist()

    generated_tokens: list[int] = []

    while True:
        prefix_len = len(generated_tokens)
        allowed_tokens = set()
        for tokens in fn_tokens.values():
            prefix = tokens[:prefix_len]
            prefix_matches = (prefix == generated_tokens)
            if prefix_matches:
                has_next_token = (len(tokens) > prefix_len)
                if has_next_token:
                    next_token = tokens[prefix_len]
                    allowed_tokens.add(next_token)

        if not allowed_tokens:
            break

        if len(allowed_tokens) == 1:
            best_token = list(allowed_tokens)[0]
        else:
            logits = np.array(
                model.get_logits_from_input_ids(
                    input_ids + generated_tokens
                )
            )
            mask = np.full(len(logits), -np.inf)
            for t_id in allowed_tokens:
                mask[t_id] = logits[t_id]
            best_token = int(np.argmax(mask))

        generated_tokens.append(best_token)

    for name, tokens in fn_tokens.items():
        if tokens == generated_tokens:
            return name

    return functions[0].name


def extract_parameters(
    model: Small_LLM_Model,
    prompt: str,
    function_def: FunctionDefinition,
) -> dict[str, Any]:
    """Extract typed arguments using constrained decoding.

    For numbers: only digit/dot/minus tokens are allowed (true masking).
    For strings: greedy decoding until closing quote.

    Args:
        model: Loaded Small_LLM_Model instance.
        prompt: The user natural language request.
        function_def: The selected FunctionDefinition schema.

    Returns:
        A dictionary mapping parameter names to their extracted values.
    """
    # 1. Pre-compute number token IDs by encoding each digit char
    #    Qwen3 tokenizes numbers char-by-char: "42" -> [token_4, token_2]
    number_tids: set[int] = set()
    for c in "0123456789.-":
        tids = model.encode(c)[0].tolist()
        number_tids.update(tids)

    stop_tids: set[int] = set()
    for c in [",", "}", " "]:
        tids = model.encode(c)[0].tolist()
        stop_tids.update(tids)

    # prefix = (
    #     f"Task: {prompt}\n"
    #     f"Function: {function_def.name}\n"
    #     f"Arguments: {{"
    # )

    prefix = (
        f"Task: {prompt}\n"
        f"Function: {function_def.name}\n"
        f"Description: {function_def.description}\n"
        f"Arguments: {{"
    )

    extracted: dict[str, Any] = {}
    params = list(function_def.parameters.items())

    for idx, (param_name, param_spec) in enumerate(params):
        if idx > 0:
            prefix += ", "

        if param_spec.type == "number":
            prefix += f'"{param_name}": '
            input_ids = model.encode(prefix)[0].tolist()
            num_chars = ""

            for _ in range(12):
                logits = np.array(
                    model.get_logits_from_input_ids(input_ids)
                )

                mask = np.full(len(logits), -np.inf)
                for t_id in number_tids | stop_tids:
                    mask[t_id] = logits[t_id]
                best_token = int(np.argmax(mask))

                if best_token in stop_tids:
                    break

                tok_text = model.decode([best_token])
                num_chars += tok_text.strip()
                input_ids.append(best_token)

            try:
                val = float(num_chars)
            except ValueError:
                val = 0.0

            extracted[param_name] = val
            prefix += str(val)

        else:
            prefix += f'"{param_name}": "'
            input_ids = model.encode(prefix)[0].tolist()
            text_val = ""

            for _ in range(50):
                logits = np.array(
                    model.get_logits_from_input_ids(input_ids)
                )
                top_tok = int(np.argmax(logits))
                tok_text = model.decode([top_tok])

                if '"' in tok_text or "\n" in tok_text:
                    break

                text_val += tok_text
                input_ids.append(top_tok)

            extracted[param_name] = text_val.strip()
            prefix += text_val.strip() + '"'

    return extracted


def decode_prompt(
    model: Small_LLM_Model,
    prompt: str,
    functions: list[FunctionDefinition],
) -> FunctionCallResult:
    """Run full constrained decoding pipeline for a single prompt.

    Args:
        model: Loaded Small_LLM_Model instance.
        prompt: Natural language user request.
        functions: List of available FunctionDefinition schemas.

    Returns:
        A validated FunctionCallResult instance.
    """
    fn_name = select_function(model, prompt, functions)
    fn_def = next(fn for fn in functions if fn.name == fn_name)
    parameters = extract_parameters(model, prompt, fn_def)

    return FunctionCallResult(
        prompt=prompt,
        name=fn_name,
        parameters=parameters,
    )
