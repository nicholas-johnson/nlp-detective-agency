"""Exercise 03 — Corpus Loader

Load Inkwell JSON data, compute word-frequency statistics, filter records.
Implement each function below — the tests in test_start.py will tell you
when you've got it right.
"""

import json  # noqa: F401
import sys  # noqa: F401
from collections import Counter, defaultdict  # noqa: F401
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell"


def load_statements() -> list[dict]:
    """Load and return the list of statement records from statements.json."""
    # TODO: implement
    pass


def extract_texts(records: list[dict]) -> list[str]:
    """Return a list of the 'raw_text' field from each record."""
    # TODO: implement
    pass


def word_frequencies(texts: list[str]) -> Counter:
    """Return a Counter of lowercased word frequencies across all *texts*.

    Words are split on whitespace (no further cleaning).
    """
    # TODO: implement
    pass


def top_words(freq: Counter, n: int = 10) -> list[tuple[str, int]]:
    """Return the *n* most common words from a frequency Counter."""
    # TODO: implement
    pass


def filter_by_case(records: list[dict], case_id: str) -> list[dict]:
    """Return only records whose 'case_id' matches *case_id*."""
    # TODO: implement
    pass


def group_by_witness(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by the 'witness' field, returning a dict mapping
    witness name to the list of their records."""
    # TODO: implement
    pass


def unique_case_ids(records: list[dict]) -> set[str]:
    """Return the set of unique case IDs across all records."""
    # TODO: implement
    pass


def summary(records: list[dict]) -> dict:
    """Return a summary dict with keys: total, cases, witnesses, top_words."""
    # TODO: implement — call the functions above
    pass


def main() -> None:
    records = load_statements()

    case_id = sys.argv[1] if len(sys.argv) > 1 else None
    if case_id:
        records = filter_by_case(records, case_id)
        if not records:
            print(f"No statements found for {case_id}")
            return

    s = summary(records)
    heading = f"Case {case_id}" if case_id else "All statements"
    print(f"\n=== {heading} ===")
    print(f"Statements: {s['total']}")
    print(f"Cases:      {', '.join(s['cases'])}")
    print(f"Witnesses:  {', '.join(s['witnesses'])}")
    print(f"\nTop words:")
    for word, count in s["top_words"]:
        print(f"  {word:<20} {count}")


if __name__ == "__main__":
    main()
