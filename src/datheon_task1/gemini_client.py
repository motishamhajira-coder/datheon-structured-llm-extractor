"""Thin wrapper around the Google Gemini API (google-genai)."""

from __future__ import annotations

import time
from typing import Callable

from google import genai

from datheon_task1.config import Settings, get_settings

DEFAULT_MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 0.5


class GeminiAPIError(Exception):
    """Gemini request failed after retries or returned unusable data."""


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in ("429", "rate limit", "unavailable", "timeout", "503")
    )


def generate_text(
    prompt: str,
    *,
    settings: Settings | None = None,
    client: genai.Client | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    """
    Send a single prompt and return the model's text response.

    Retries transient network/rate-limit errors up to max_retries times.
    """
    settings = settings or get_settings()
    client = client or create_client(settings)
    last_error: BaseException | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            text = response.text
            if text is None or not text.strip():
                raise GeminiAPIError("Gemini returned an empty response.")
            return text.strip()
        except GeminiAPIError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries or not _is_retryable_error(exc):
                break
            sleep_fn(RETRY_BACKOFF_SECONDS * attempt)

    message = f"Gemini API call failed after {max_retries} attempt(s)."
    if last_error is not None:
        message = f"{message} Last error: {last_error}"
    raise GeminiAPIError(message) from last_error


def create_client(settings: Settings | None = None) -> genai.Client:
    """Build a Gemini client using the API key from settings."""
    settings = settings or get_settings()
    api_key = settings.require_api_key()
    return genai.Client(api_key=api_key)
