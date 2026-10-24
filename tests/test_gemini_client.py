"""Mocked tests for the Gemini client wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from datheon_task1.config import Settings
from datheon_task1 import gemini_client
from datheon_task1.gemini_client import GeminiAPIError


def test_generate_text_uses_client_and_returns_stripped_text() -> None:
    settings = Settings(gemini_api_key="fake-key", gemini_model="gemini-mock")
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "  OK  "
    mock_client.models.generate_content.return_value = mock_response

    result = gemini_client.generate_text(
        "hello",
        settings=settings,
        client=mock_client,
    )

    assert result == "OK"
    mock_client.models.generate_content.assert_called_once_with(
        model="gemini-mock",
        contents="hello",
    )


def test_create_client_passes_api_key(monkeypatch) -> None:
    settings = Settings(gemini_api_key="my-secret", gemini_model="gemini-mock")
    captured: dict = {}

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            captured["api_key"] = api_key

    monkeypatch.setattr(gemini_client.genai, "Client", FakeClient)

    gemini_client.create_client(settings)

    assert captured["api_key"] == "my-secret"


def test_generate_text_retries_then_succeeds() -> None:
    settings = Settings(gemini_api_key="fake-key", gemini_model="gemini-mock")
    mock_client = MagicMock()
    ok_response = MagicMock(text="done")
    mock_client.models.generate_content.side_effect = [
        ConnectionError("temporary"),
        ok_response,
    ]
    sleeps: list[float] = []

    result = gemini_client.generate_text(
        "hi",
        settings=settings,
        client=mock_client,
        max_retries=3,
        sleep_fn=sleeps.append,
    )

    assert result == "done"
    assert mock_client.models.generate_content.call_count == 2
    assert sleeps == [0.5]


def test_generate_text_raises_gemini_api_error_when_exhausted() -> None:
    settings = Settings(gemini_api_key="fake-key", gemini_model="gemini-mock")
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = ConnectionError("still down")

    with pytest.raises(GeminiAPIError, match="failed after"):
        gemini_client.generate_text(
            "hi",
            settings=settings,
            client=mock_client,
            max_retries=2,
            sleep_fn=lambda _: None,
        )


def test_generate_text_empty_response_raises_gemini_api_error() -> None:
    settings = Settings(gemini_api_key="fake-key", gemini_model="gemini-mock")
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = MagicMock(text="  ")

    with pytest.raises(GeminiAPIError, match="empty"):
        gemini_client.generate_text("hi", settings=settings, client=mock_client)
