"""Mocked tests for the full extraction pipeline."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from datheon_task1.config import Settings
from datheon_task1.extractor import JSONParseError
from datheon_task1.gemini_client import GeminiAPIError
from datheon_task1.pipeline import extract_invoice_from_text


def test_pipeline_happy_path() -> None:
    settings = Settings(gemini_api_key="key", gemini_model="gemini-test")
    mock_client = MagicMock()
    payload = {"vendor_name": "Pipeline Co", "total": "50.00"}
    mock_client.models.generate_content.return_value = MagicMock(
        text=json.dumps(payload),
    )

    invoice = extract_invoice_from_text(
        "Invoice from Pipeline Co total 50",
        settings=settings,
        client=mock_client,
        examples=[],
    )

    assert invoice.vendor_name == "Pipeline Co"
    assert str(invoice.total) == "50.00"
    call = mock_client.models.generate_content.call_args
    assert call.kwargs["model"] == "gemini-test"
    assert "Pipeline Co total 50" in call.kwargs["contents"]


def test_pipeline_api_error_propagates() -> None:
    settings = Settings(gemini_api_key="key", gemini_model="gemini-test")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = ConnectionError("network down")

    with pytest.raises(GeminiAPIError):
        extract_invoice_from_text("text", settings=settings, client=mock_client, examples=[])


def test_pipeline_extraction_error_on_bad_json() -> None:
    settings = Settings(gemini_api_key="key", gemini_model="gemini-test")
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text="not json at all")

    with pytest.raises(JSONParseError):
        extract_invoice_from_text("text", settings=settings, client=mock_client, examples=[])
