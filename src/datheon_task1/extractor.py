"""Parse and validate JSON invoice data from raw LLM text."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from datheon_task1.schemas import Invoice, parse_invoice

_FENCE_FULL = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL | re.IGNORECASE,
)


class ExtractionError(Exception):
    """Could not turn LLM text into a valid Invoice."""


class JSONParseError(ExtractionError):
    """The response did not contain parseable JSON object."""


class InvoiceValidationError(ExtractionError):
    """JSON parsed but did not match the Invoice schema."""

    def __init__(self, message: str, errors: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


def strip_markdown_fence(text: str) -> str:
    """Remove ```json ... ``` wrappers if present."""
    stripped = text.strip()
    match = _FENCE_FULL.match(stripped)
    if match:
        return match.group(1).strip()

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()

    return stripped


def _load_json_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as first_error:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise JSONParseError(
                "Response is not valid JSON. Check for extra text or typos."
            ) from first_error
        snippet = text[start : end + 1]
        try:
            data = json.loads(snippet)
        except json.JSONDecodeError as second_error:
            raise JSONParseError(
                "Found a JSON-like block but could not parse it."
            ) from second_error

    if not isinstance(data, dict):
        raise JSONParseError(
            f"Expected one JSON object, got {type(data).__name__} instead."
        )
    return data


def extract_json_object(raw_text: str) -> dict[str, Any]:
    """
    Extract a JSON object dict from model output.

    Handles markdown fences and extra prose before/after the object.
    """
    if not raw_text or not raw_text.strip():
        raise JSONParseError("Model returned an empty response.")

    cleaned = strip_markdown_fence(raw_text)
    return _load_json_object(cleaned)


def _format_validation_errors(errors: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for err in errors:
        loc = ".".join(str(x) for x in err.get("loc", ()))
        msg = err.get("msg", "invalid value")
        if loc:
            parts.append(f"{loc}: {msg}")
        else:
            parts.append(str(msg))
    return "; ".join(parts) if parts else "Schema validation failed."


def extract_invoice_from_llm(raw_text: str) -> Invoice:
    """Full pipeline: LLM string → dict → validated Invoice."""
    data = extract_json_object(raw_text)
    try:
        return parse_invoice(data)
    except PydanticValidationError as exc:
        errors = list(exc.errors())
        message = _format_validation_errors(errors)
        raise InvoiceValidationError(message, errors=errors) from exc
