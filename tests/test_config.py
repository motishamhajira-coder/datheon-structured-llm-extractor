"""Tests for environment configuration (no live API)."""

from __future__ import annotations

import pytest

from datheon_task1.config import DEFAULT_GEMINI_MODEL, Settings, get_settings


def test_get_settings_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test-model")

    settings = get_settings()

    assert settings.gemini_api_key == "test-key-123"
    assert settings.gemini_model == "gemini-test-model"


def test_default_model_when_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k")

    settings = get_settings()

    assert settings.gemini_model == DEFAULT_GEMINI_MODEL


def test_require_api_key_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    settings = get_settings()

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        settings.require_api_key()


def test_require_api_key_strips_whitespace() -> None:
    settings = Settings(gemini_api_key="  abc  ", gemini_model=DEFAULT_GEMINI_MODEL)
    assert settings.require_api_key() == "abc"
