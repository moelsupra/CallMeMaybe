"""Data schemas, JSON loader, and error handling for CallMeMaybe."""

import json
import sys
from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, StringConstraints, ValidationError

CleanStr = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1)
]

DataType = Literal["number", "string", "integer", "boolean"]


class StrictBaseModel(BaseModel):
    """Base model that forbids unexpected extra keys in input JSON."""

    model_config = ConfigDict(extra="forbid")


class PromptTest(StrictBaseModel):
    """Schema representing an input test prompt.

    Attributes:
        prompt: Natural language user request.
    """

    prompt: CleanStr


class ParameterDefinition(StrictBaseModel):
    """Schema representing a function parameter type specification.

    Attributes:
        type: The allowed data type ('number', 'string', 'integer', 'boolean').
    """

    type: DataType


class FunctionDefinition(StrictBaseModel):
    """Schema representing a callable function specification.

    Attributes:
        name: Name of the function.
        description: Description of the function purpose.
        parameters: Mapping of parameter names to ParameterDefinition.
        returns: Return type specification.
    """

    name: CleanStr
    description: CleanStr
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition


class FunctionCallResult(StrictBaseModel):
    """Schema representing the structured output of a function call.

    Attributes:
        prompt: The original user request.
        name: The selected function name.
        parameters: Extracted argument values.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]


def raise_on_duplicate_keys(
    ordered_pairs: list[tuple[str, Any]]
) -> dict[str, Any]:
    """Reject duplicate keys during JSON object decoding.

    Args:
        ordered_pairs: List of key-value pairs parsed in source order.

    Returns:
        dict[str, Any]: Dictionary containing unique keys and values.

    Raises:
        ValueError: If a duplicate key is encountered.
    """
    result: dict[str, Any] = {}
    for key, value in ordered_pairs:
        if key in result:
            raise ValueError(f"Duplicate key found in JSON: '{key}'")
        result[key] = value
    return result


def load_function_definitions(path: str) -> list[FunctionDefinition]:
    """Load and validate function definitions from a JSON file.

    Args:
        path: Path to the function definitions JSON file.

    Returns:
        list[FunctionDefinition]: List of validated FunctionDefinition models.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the JSON syntax is invalid.
        ValueError: If duplicate keys exist, not a list, empty,
            or invalid identifiers.
        ValidationError: If fields do not match FunctionDefinition schema.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f, object_pairs_hook=raise_on_duplicate_keys)

    if not isinstance(data, list) or not data:
        raise ValueError(
            f"File '{path}' must contain a non-empty list of functions."
        )

    functions = [FunctionDefinition.model_validate(item) for item in data]
    seen_names: set[str] = set()

    for fn in functions:
        if fn.name in seen_names:
            raise ValueError(
                f"Duplicate function definition: '{fn.name}'."
            )
        seen_names.add(fn.name)

        if not fn.name.isidentifier():
            raise ValueError(
                f"Function name '{fn.name}' is not a valid identifier."
            )
        for param_name in fn.parameters:
            if not param_name.isidentifier():
                raise ValueError(
                    f"Parameter '{param_name}' in function '{fn.name}' "
                    f"is not a valid identifier."
                )

    return functions


def load_prompt_tests(path: str) -> list[PromptTest]:
    """Load and validate prompt tests from a JSON file.

    Args:
        path: Path to the input test prompts JSON file.

    Returns:
        list[PromptTest]: List of validated PromptTest models.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the JSON syntax is invalid.
        ValueError: If duplicate keys exist, not a list, or empty.
        ValidationError: If fields do not match PromptTest schema.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f, object_pairs_hook=raise_on_duplicate_keys)

    if not isinstance(data, list) or not data:
        raise ValueError(
            f"File '{path}' must contain a non-empty list of prompts."
        )

    return [PromptTest.model_validate(item) for item in data]


def handle_error(exc: Exception) -> None:
    """Format user-facing errors cleanly and exit with status code 1.

    Args:
        exc: Exception raised during application execution.
    """
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
            err_type = err.get("type", "")

            if err_type == "extra_forbidden":
                print(
                    f"[ERROR] Unexpected key '{field}' is not allowed.",
                    file=sys.stderr,
                )
            elif err_type == "missing":
                print(
                    f"[ERROR] Missing required field '{field}'.",
                    file=sys.stderr,
                )
            elif (
                "cannot be empty" in err["msg"]
                or err_type == "string_too_short"
            ):
                print(
                    f"[ERROR] Field '{field}' cannot be empty.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[ERROR] Invalid value for '{field}': {err['msg']}",
                    file=sys.stderr,
                )
    else:
        print(f"[ERROR] {exc}", file=sys.stderr)

    sys.exit(1)
