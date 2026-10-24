"""End-to-end flow: unstructured text → prompt → Gemini → validated Invoice."""

from __future__ import annotations

from google import genai

from datheon_task1.config import Settings, get_settings
from datheon_task1.extractor import ExtractionError, extract_invoice_from_llm
from datheon_task1.gemini_client import GeminiAPIError, generate_text
from datheon_task1.prompts import FewShotExample, build_invoice_extraction_prompt
from datheon_task1.schemas import Invoice


def extract_invoice_from_text(
    invoice_text: str,
    *,
    settings: Settings | None = None,
    client: genai.Client | None = None,
    examples: tuple[FewShotExample, ...] | list[FewShotExample] | None = None,
    extra_instructions: str = "",
) -> Invoice:
    """
    Build the extraction prompt, call Gemini, parse JSON, validate with Pydantic.

    Raises:
        GeminiAPIError: API/key/network issues or empty model output.
        ExtractionError: JSON parse or schema validation failures.
    """
    prompt = build_invoice_extraction_prompt(
        invoice_text,
        examples=examples,
        extra_instructions=extra_instructions,
    )
    raw_response = generate_text(
        prompt,
        settings=settings,
        client=client,
    )
    return extract_invoice_from_llm(raw_response)
