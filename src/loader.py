"""Load and validate input JSON files for CallMeMaybe."""

import json
from src.models import FunctionDefinition, PromptTest


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the JSON syntax is invalid.
        ValueError: If the file is not a list or is empty.
        pydantic.ValidationError: If items do not match FunctionDefinition.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(
            f"File '{path}' must contain a non-empty list of functions."
        )

    return [FunctionDefinition.model_validate(item) for item in data]


def load_prompt_tests(path: str) -> list[PromptTest]:
    """Load and validate prompt tests from a JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the JSON syntax is invalid.
        ValueError: If the file is not a list or is empty.
        pydantic.ValidationError: If items do not match PromptTest.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(
            f"File '{path}' must contain a non-empty list of prompts."
        )

    return [PromptTest.model_validate(item) for item in data]
