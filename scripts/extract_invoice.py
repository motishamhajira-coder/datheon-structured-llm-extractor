import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from datheon_task1.pipeline import extract_invoice_from_text


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured invoice data using Gemini."
    )
    parser.add_argument("file", help="Path to an invoice text file")
    args = parser.parse_args()

    load_dotenv()

    try:
        text = Path(args.file).read_text(encoding="utf-8")
        invoice = extract_invoice_from_text(text)
        print(json.dumps(invoice.model_dump(mode="json"), indent=2))
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())