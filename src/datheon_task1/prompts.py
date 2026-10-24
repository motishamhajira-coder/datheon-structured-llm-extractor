"""Parameterized prompt templates and few-shot examples for invoice extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from datheon_task1.schemas import invoice_json_schema

# --- Templates (placeholders: {role_instructions}, {json_schema}, {few_shot_block}, {invoice_text}) ---

ROLE_INSTRUCTIONS = """You are an invoice data extraction assistant.
Read unstructured invoice text and return ONE JSON object that matches the schema below.
Use null for fields that are not present in the text. Do not invent values.
For money and quantities, use JSON numbers or numeric strings.
Return JSON only — no markdown fences, no commentary."""

FEW_SHOT_EXAMPLE_HEADER = "Few-shot examples (input text → JSON output):"

EXAMPLE_BLOCK_TEMPLATE = """\
Example {index}:
Input:
{input_text}
Output:
{output_json}"""

PROMPT_TEMPLATE = """\
{role_instructions}

{extra_instructions}JSON Schema for the output object:
{json_schema}

{few_shot_block}
Invoice text to extract:
---
{invoice_text}
---
"""


@dataclass(frozen=True)
class FewShotExample:
    """One demonstration pair for the prompt."""

    input_text: str
    output: dict[str, Any]


# Default examples: small, realistic, and valid against Invoice schema.
DEFAULT_FEW_SHOT_EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample(
        input_text=(
            "INVOICE #A-42\n"
            "From: Bright Office LLC\n"
            "Bill to: Sam Rivera\n"
            "Date: 2024-01-10\n"
            "Item: Desk lamp  Qty 1  $45.00\n"
            "Total: $45.00 USD"
        ),
        output={
            "invoice_number": "A-42",
            "invoice_date": "2024-01-10",
            "vendor_name": "Bright Office LLC",
            "customer_name": "Sam Rivera",
            "currency": "USD",
            "line_items": [
                {
                    "description": "Desk lamp",
                    "quantity": 1,
                    "unit_price": "45.00",
                    "amount": "45.00",
                }
            ],
            "total": "45.00",
        },
    ),
    FewShotExample(
        input_text=(
            "Tax invoice\n"
            "Supplier: Northwind Traders\n"
            "Description: Monthly support\n"
            "Amount due EUR 120.00"
        ),
        output={
            "vendor_name": "Northwind Traders",
            "currency": "EUR",
            "line_items": [
                {"description": "Monthly support", "amount": "120.00"},
            ],
            "total": "120.00",
        },
    ),
)


def _format_json_for_prompt(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def render_few_shot_block(
    examples: tuple[FewShotExample, ...] | list[FewShotExample],
) -> str:
    """Build the few-shot section; empty string if there are no examples."""
    if not examples:
        return ""

    blocks = [
        EXAMPLE_BLOCK_TEMPLATE.format(
            index=i + 1,
            input_text=ex.input_text.strip(),
            output_json=_format_json_for_prompt(ex.output),
        )
        for i, ex in enumerate(examples)
    ]
    return FEW_SHOT_EXAMPLE_HEADER + "\n\n" + "\n\n".join(blocks) + "\n\n"


def build_invoice_extraction_prompt(
    invoice_text: str,
    *,
    examples: tuple[FewShotExample, ...] | list[FewShotExample] | None = None,
    extra_instructions: str = "",
    role_instructions: str = ROLE_INSTRUCTIONS,
) -> str:
    """
    Build the full user prompt for Gemini.

    Parameters let you swap few-shot sets or add task-specific rules without
    editing the core template.
    """
    if examples is None:
        examples = DEFAULT_FEW_SHOT_EXAMPLES

    schema_str = _format_json_for_prompt(invoice_json_schema())
    few_shot_block = render_few_shot_block(examples)

    extra = extra_instructions.strip()
    if extra:
        extra = extra + "\n\n"

    return PROMPT_TEMPLATE.format(
        role_instructions=role_instructions.strip(),
        extra_instructions=extra,
        json_schema=schema_str,
        few_shot_block=few_shot_block,
        invoice_text=invoice_text.strip(),
    ).strip()
