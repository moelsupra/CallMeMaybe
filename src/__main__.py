"""Main entry point for CallMeMaybe."""

import argparse
import json
import sys

from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from pydantic import ValidationError
from src.loader import load_function_definitions, load_prompt_tests
from src.decoder import decode_prompt


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


def handle_error(exc: Exception) -> None:
    """Format user-facing errors cleanly and exit with status code 1."""
    if isinstance(exc, FileNotFoundError):
        print(f"[ERROR] File not found: '{exc.filename}'", file=sys.stderr)
    elif isinstance(exc, json.JSONDecodeError):
        print(
            f"[ERROR] Invalid JSON syntax: {exc.msg} (line {exc.lineno})",
            file=sys.stderr,
        )
    elif isinstance(exc, ValidationError):
        for err in exc.errors():
            field = " -> ".join(str(k) for k in err["loc"])
            print(
                f"[ERROR] Schema error on '{field}': {err['msg']}",
                file=sys.stderr,
            )
    else:
        print(f"[ERROR] {exc}", file=sys.stderr)

    sys.exit(1)


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
        for prompt_test in prompts:
            result = decode_prompt(model, prompt_test.prompt, functions)
            results.append(result.model_dump())
            print(f"  ✅ {result.name}({result.parameters})")

    except Exception as exc:
        handle_error(exc)


if __name__ == "__main__":
    main()
