# Structured LLM Prompt Pipeline & JSON Extractor

A Python project for extracting structured invoice data from unstructured text using Google Gemini and validating the result with Pydantic.

## Features

- Gemini API integration using `google-genai`
- Parameterized prompt templates and few-shot examples
- Pydantic invoice schema validation
- JSON parsing and validation error handling
- Retry handling for transient API failures
- Command-line interface for invoice text files
- Automated tests using mocked responses

## Requirements

- Python 3.10+
- Google Gemini API key for live extraction

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your Gemini API key:

```text
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash
```

Never commit `.env` or expose your API key.

## Usage

Run invoice extraction:

```powershell
$env:PYTHONPATH = "src"
python scripts/extract_invoice.py samples/sample_invoice.txt
```

The CLI prints validated JSON on success and an error message on failure. Live extraction requires a valid API key and an accessible Gemini model.

## Tests

Run the automated test suite without making live API calls:

```powershell
python -m pytest -q
```

## Project Structure

```text
TASK1/
├── samples/
│   └── sample_invoice.txt
├── scripts/
│   ├── extract_invoice.py
│   └── smoke_gemini.py
├── src/datheon_task1/
├── tests/
├── .env.example
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Limitations

Extraction accuracy depends on the source text and LLM response. Pydantic validates data structure and types; it does not independently verify invoice totals or the truth of extracted information.