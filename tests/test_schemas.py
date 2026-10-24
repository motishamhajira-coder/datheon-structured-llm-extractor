"""Tests for invoice Pydantic schemas (no API)."""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from datheon_task1.schemas import Invoice, LineItem, invoice_json_schema, parse_invoice


def test_parse_full_invoice() -> None:
    payload = {
        "invoice_number": "INV-1001",
        "invoice_date": "2024-03-15",
        "due_date": "2024-04-15",
        "vendor_name": "Acme Supplies",
        "vendor_address": "123 Main St",
        "customer_name": "Jane Doe",
        "currency": "usd",
        "line_items": [
            {
                "description": "Widget",
                "quantity": "2",
                "unit_price": "10.50",
                "amount": "21.00",
            }
        ],
        "subtotal": "21.00",
        "tax": "1.68",
        "total": "22.68",
    }

    invoice = parse_invoice(payload)

    assert invoice.invoice_number == "INV-1001"
    assert invoice.currency == "USD"
    assert len(invoice.line_items) == 1
    assert invoice.line_items[0].quantity == Decimal("2")
    assert invoice.total == Decimal("22.68")


def test_minimal_invoice_allows_missing_fields() -> None:
    invoice = parse_invoice({"vendor_name": "Solo Vendor"})

    assert invoice.vendor_name == "Solo Vendor"
    assert invoice.invoice_number is None
    assert invoice.line_items == []
    assert invoice.total is None


def test_empty_object_is_valid() -> None:
    invoice = parse_invoice({})

    assert isinstance(invoice, Invoice)
    assert invoice.line_items == []


def test_strips_string_whitespace() -> None:
    invoice = parse_invoice({"invoice_number": "  INV-9  ", "vendor_name": "  Co  "})

    assert invoice.invoice_number == "INV-9"
    assert invoice.vendor_name == "Co"


def test_line_item_partial_fields() -> None:
    item = LineItem.model_validate({"description": "Consulting"})
    assert item.description == "Consulting"
    assert item.quantity is None


def test_invalid_line_item_type_raises() -> None:
    with pytest.raises(ValidationError):
        parse_invoice({"line_items": [{"quantity": "not-a-number"}]})


def test_invoice_json_schema_has_invoice_title() -> None:
    schema = invoice_json_schema()
    assert schema.get("title") == "Invoice"
    assert "line_items" in schema.get("properties", {})
