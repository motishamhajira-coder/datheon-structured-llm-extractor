"""Pydantic models for structured invoice data extracted from text."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LineItem(BaseModel):
    """One row on an invoice (all fields optional when the source text omits them)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    description: str | None = None
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal | None = None


class Invoice(BaseModel):
    """
    Structured invoice returned by the LLM as JSON.

    Required fields are kept minimal so partial extractions still validate.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    invoice_number: str | None = None
    invoice_date: str | None = Field(
        default=None,
        description="ISO-like date string as printed on the invoice (e.g. 2024-03-15).",
    )
    due_date: str | None = None
    vendor_name: str | None = None
    vendor_address: str | None = None
    customer_name: str | None = None
    currency: str | None = Field(default=None, description="ISO 4217 code, e.g. USD")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Decimal | None = None
    tax: Decimal | None = None
    total: Decimal | None = None

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().upper()
        return cleaned or None


def parse_invoice(data: dict[str, Any]) -> Invoice:
    """Validate a JSON object and return an Invoice model."""
    return Invoice.model_validate(data)


def invoice_json_schema() -> dict[str, Any]:
    """JSON Schema for prompts and documentation (used in later stages)."""
    return Invoice.model_json_schema()
