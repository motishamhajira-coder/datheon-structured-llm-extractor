import os
import subprocess
import sys
from pathlib import Path


def test_cli_help():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/extract_invoice.py", "--help"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "invoice text file" in result.stdout.lower()