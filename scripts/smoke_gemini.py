"""Optional live check: one short request to Gemini (needs GEMINI_API_KEY)."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as: python scripts/smoke_gemini.py
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datheon_task1.config import get_settings  # noqa: E402
from datheon_task1.gemini_client import generate_text  # noqa: E402


def main() -> int:
    settings = get_settings()
    try:
        settings.require_api_key()
    except ValueError as exc:
        print(exc)
        return 1

    reply = generate_text("Reply with exactly: OK", settings=settings)
    print(f"Model ({settings.gemini_model}) replied: {reply}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
