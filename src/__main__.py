"""Main entry point for CallMeMaybe."""

import argparse


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
    _ = args


if __name__ == "__main__":
    main()
