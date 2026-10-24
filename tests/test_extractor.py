"""Tests for JSON extraction and invoice validation from LLM text."""

from __future__ import annotations

import json

import pytest

from datheon_task1.extractor import (
    ExtractionError,
    InvoiceValidationError,
    JSONParseError,
    extract_invoice_from_llm,
    extract_json_object,
    strip_markdown_fence,
)
from datheon_task1.schemas import Invoice


def test_strip_markdown_fence_json_tag() -> None:
    raw = '```json\n{"vendor_name": "A"}\n```'
    assert strip_markdown_fence(raw) == '{"vendor_name": "A"}'


def test_extract_plain_json_object() -> None:
    data = extract_json_object('{"invoice_number": "1", "total": "9.99"}')
    assert data["invoice_number"] == "1"


def test_extract_json_with_surrounding_prose() -> None:
    raw = 'Here is the data:\n{"vendor_name": "Co"}\nThanks!'
    data = extract_json_object(raw)
    assert data["vendor_name"] == "Co"


def test_extract_from_markdown_fence() -> None:
    payload = {"currency": "USD", "total": "10"}
    raw = f"```json\n{json.dumps(payload)}\n```"
    invoice = extract_invoice_from_llm(raw)
    assert invoice.currency == "USD"
    assert str(invoice.total) == "10"


def test_extract_invoice_full_pipeline() -> None:
    raw = json.dumps(
        {
            "vendor_name": "Test Vendor",
            "line_items": [{"description": "Item", "amount": "5"}],
        }
    )
    invoice = extract_invoice_from_llm(raw)
    assert isinstance(invoice, Invoice)
    assert invoice.vendor_name == "Test Vendor"
    assert len(invoice.line_items) == 1


def test_empty_response_raises_json_parse_error() -> None:
    with pytest.raises(JSONParseError, match="empty"):
        extract_json_object("   ")


def test_invalid_json_raises() -> None:
    with pytest.raises(JSONParseError):
        extract_json_object("{not json}")


def test_json_array_raises() -> None:
    with pytest.raises(JSONParseError, match="object"):
        extract_json_object("[1, 2, 3]")


def test_schema_validation_error_on_bad_field_type() -> None:
    with pytest.raises(InvoiceValidationError) as exc_info:
        extract_invoice_from_llm('{"line_items": [{"quantity": "many"}]}')
    assert exc_info.value.errors
    assert isinstance(exc_info.value, ExtractionError)


def test_missing_fields_still_valid() -> None:
    invoice = extract_invoice_from_llm("{}")
    assert invoice.vendor_name is None
    assert invoice.line_items == []
