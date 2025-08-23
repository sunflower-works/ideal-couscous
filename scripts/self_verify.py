#!/usr/bin/env python3
"""
Project Sunflower – self-verification helper.

For now it only checks that the PDF exists and can be opened.  In later
phases we will parse object streams to locate the invisible U+200B
characters and tally them against the disclosure page.
"""
from pathlib import Path
import sys

try:
    import PyPDF2
except ImportError:
    print("ERROR: PyPDF2 not installed.  Run:  pip install PyPDF2>=3.0.0")
    sys.exit(1)


def verify_document_watermarks(pdf_path: Path) -> None:
    print("=" * 34)
    print(" PROJECT SUNFLOWER VERIFICATION")
    print("=" * 34)
    print(f"Analysing: {pdf_path}\n")

    if not pdf_path.exists():
        print(f"✗ ERROR: file not found: {pdf_path}")
        sys.exit(1)

    # Simple open-test – more intelligence will be added later
    try:
        with pdf_path.open("rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            print(f"✓ PDF opened, {len(reader.pages)} page(s) detected.")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"✗ ERROR: unable to parse PDF – {exc}")
        sys.exit(1)

    print("\n--- Verification complete (phase-1 stub) ---")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/self_verify.py <thesis.pdf>")
        sys.exit(1)
    verify_document_watermarks(Path(sys.argv[1]))