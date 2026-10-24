"""Load settings from environment variables and optional .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"


def _project_root() -> Path:
    """Directory that contains .env (project root), not the package folder."""
    return Path(__file__).resolve().parents[2]


def load_env() -> None:
    """Load variables from .env in the project root if the file exists."""
    env_path = _project_root() / ".env"
    load_dotenv(env_path)


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    gemini_model: str

    def require_api_key(self) -> str:
        if not self.gemini_api_key or not self.gemini_api_key.strip():
            raise ValueError(
                "GEMINI_API_KEY is missing. Copy .env.example to .env and set your key."
            )
        return self.gemini_api_key.strip()


def get_settings() -> Settings:
    """Return current settings (loads .env from project root if present)."""
    load_env()

    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    key = os.getenv("GEMINI_API_KEY")
    return Settings(gemini_api_key=key, gemini_model=model or DEFAULT_GEMINI_MODEL)
