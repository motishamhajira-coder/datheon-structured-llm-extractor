"""Tests for prompt templates and few-shot rendering."""

from __future__ import annotations

import json

import pytest

from datheon_task1.prompts import (
    DEFAULT_FEW_SHOT_EXAMPLES,
    FewShotExample,
    build_invoice_extraction_prompt,
    render_few_shot_block,
)
from datheon_task1.schemas import parse_invoice


def test_build_prompt_includes_invoice_text() -> None:
    text = "INVOICE 999\nTotal $10"
    prompt = build_invoice_extraction_prompt(text, examples=[])

    assert "INVOICE 999" in prompt
    assert "Total $10" in prompt
    assert "Invoice text to extract:" in prompt


def test_build_prompt_includes_json_schema() -> None:
    prompt = build_invoice_extraction_prompt("x", examples=[])

    assert '"title": "Invoice"' in prompt or '"Invoice"' in prompt
    assert "line_items" in prompt


def test_default_few_shot_examples_in_prompt() -> None:
    prompt = build_invoice_extraction_prompt("Sample invoice body")

    assert "Few-shot examples" in prompt
    assert "Bright Office LLC" in prompt
    assert "Northwind Traders" in prompt
    assert "Example 1:" in prompt
    assert "Example 2:" in prompt


def test_custom_few_shot_example_appears() -> None:
    custom = (
        FewShotExample(
            input_text="Custom input line",
            output={"vendor_name": "Custom Co"},
        ),
    )
    prompt = build_invoice_extraction_prompt("target", examples=custom)

    assert "Custom input line" in prompt
    assert "Custom Co" in prompt
    assert "Bright Office LLC" not in prompt


def test_extra_instructions_inserted() -> None:
    prompt = build_invoice_extraction_prompt(
        "inv",
        examples=[],
        extra_instructions="Always use YYYY-MM-DD for dates.",
    )

    assert "Always use YYYY-MM-DD for dates." in prompt


def test_render_few_shot_block_empty() -> None:
    assert render_few_shot_block([]) == ""


def test_default_few_shot_outputs_validate_with_schema() -> None:
    for ex in DEFAULT_FEW_SHOT_EXAMPLES:
        parse_invoice(ex.output)


def test_few_shot_output_is_valid_json_in_prompt() -> None:
    examples = (
        FewShotExample(
            input_text="T",
            output={"invoice_number": "1"},
        ),
    )
    block = render_few_shot_block(examples)
    # Extract JSON after "Output:" for a sanity check
    json_part = block.split("Output:\n", 1)[1].strip()
    parsed = json.loads(json_part)
    assert parsed["invoice_number"] == "1"
