"""Main entry point for CallMeMaybe."""

import argparse
import json
import os
from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.loader import (
    handle_error,
    load_function_definitions,
    load_prompt_tests,
)
from src.decoder import decode_prompt
import time


def parse_args_cli() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Translate natural language prompts into function calls "
            "using constrained decoding."
        )
    )

    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input tests JSON file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the CallMeMaybe application."""
    print("🤖 Starting CallMeMaybe...")
    args = parse_args_cli()

    try:
        functions = load_function_definitions(args.functions_definition)
        prompts = load_prompt_tests(args.input)
        print(f"Loaded {len(functions)} functions and {len(prompts)} prompts.")

        model = Small_LLM_Model()
        results: list[dict[str, object]] = []
        start = time.perf_counter()
        for prompt_test in prompts:
            result = decode_prompt(model, prompt_test.prompt, functions)
            results.append(result.model_dump())
            print(f"  ✅ {result.name}({result.parameters})")

        end = time.perf_counter()
        min, sec = divmod(end-start, 60)
        print(f"generation time: {int(min)}m {sec:.2f}s")
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\n📄 Output written to {args.output}")

    except Exception as exc:
        handle_error(exc)


if __name__ == "__main__":
    main()
