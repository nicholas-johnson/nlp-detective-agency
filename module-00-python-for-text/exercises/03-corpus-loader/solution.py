"""Exercise 03 — Corpus Loader (solution)

Load Inkwell JSON data, compute word-frequency statistics, filter records.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell"


def load_statements() -> list[dict]:
    """Load and return the list of statement records from statements.json."""
    path = DATA_DIR / "statements.json"
    return json.loads(path.read_text())


def extract_texts(records: list[dict]) -> list[str]:
    """Return a list of the 'raw_text' field from each record."""
    return [r["raw_text"] for r in records]


def word_frequencies(texts: list[str]) -> Counter:
    """Return a Counter of lowercased word frequencies across all *texts*.

    Words are split on whitespace (no further cleaning).
    """
    return Counter(w.lower() for text in texts for w in text.split())


def top_words(freq: Counter, n: int = 10) -> list[tuple[str, int]]:
    """Return the *n* most common words from a frequency Counter."""
    return freq.most_common(n)


def filter_by_case(records: list[dict], case_id: str) -> list[dict]:
    """Return only records whose 'case_id' matches *case_id*."""
    return [r for r in records if r["case_id"] == case_id]


def group_by_witness(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by the 'witness' field, returning a dict mapping
    witness name to the list of their records."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["witness"]].append(r)
    return dict(groups)


def unique_case_ids(records: list[dict]) -> set[str]:
    """Return the set of unique case IDs across all records."""
    return {r["case_id"] for r in records}


def summary(records: list[dict]) -> dict:
    """Return a summary dict with keys: total, cases, witnesses, top_words."""
    texts = extract_texts(records)
    freq = word_frequencies(texts)
    return {
        "total": len(records),
        "cases": sorted(unique_case_ids(records)),
        "witnesses": sorted({r["witness"] for r in records}),
        "top_words": top_words(freq, 5),
    }


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
