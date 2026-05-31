"""Demo 01 — Strings and Regex on Inkwell Data

Interactive walkthrough of string methods and regex applied to real witness
statements. Run with:

    python module-00-python-for-text/demo/01_strings_and_regex.py
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "inkwell"


def main() -> None:
    statements = json.loads((DATA_DIR / "statements.json").read_text())
    raw = statements[0]["raw_text"]

    print("=" * 60)
    print("DEMO 1 — Strings and Regex")
    print("=" * 60)

    # --- String basics ---
    print("\n--- Raw statement ---")
    print(repr(raw))

    print("\n--- .strip().lower() ---")
    print(raw.strip().lower())

    print("\n--- .split() (whitespace tokens) ---")
    tokens = raw.split()
    print(tokens)
    print(f"Token count: {len(tokens)}")

    print("\n--- Slicing: first 40 characters ---")
    print(raw[:40])

    # --- f-strings ---
    witness = statements[0]["witness"]
    case_id = statements[0]["case_id"]
    print(f"\n--- f-string ---")
    print(f"Witness: {witness} | Case: {case_id} | Words: {len(tokens)}")

    # --- Regex ---
    print("\n--- Regex: find case IDs ---")
    case_ids = re.findall(r"CASE-\d+", raw)
    print(f"Found: {case_ids}")

    print("\n--- Regex: find [REDACTED] markers ---")
    redacted = re.findall(r"\[REDACTED\]", raw)
    print(f"Found: {redacted}")

    print("\n--- Regex: strip non-alpha characters ---")
    cleaned = re.sub(r"[^a-zA-Z\s]", "", raw)
    print(cleaned)

    print("\n--- Regex: count ALL-CAPS words (2+ letters) ---")
    caps = re.findall(r"\b[A-Z]{2,}\b", raw)
    print(f"Uppercase words: {caps}")

    # --- All statements ---
    print("\n--- All statements: dates found ---")
    for s in statements:
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", s.get("recorded", ""))
        if dates:
            print(f"  {s['id']}: {dates}")

    print("\nDone.")


if __name__ == "__main__":
    main()
