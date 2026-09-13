"""Data schemas and Pydantic validation models for CallMeMaybe."""

from typing import Any
from pydantic import BaseModel


class PromptTest(BaseModel):
    """Input prompt test case."""

    prompt: str


class ParameterDefinition(BaseModel):
    """Specification of a parameter type."""

    type: str


class FunctionDefinition(BaseModel):
    """Specification of a callable function."""

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: dict[str, str] | None = None


class FunctionCallResult(BaseModel):
    """Result of a function call."""

    prompt: str
    name: str
    parameters: dict[str, Any]

