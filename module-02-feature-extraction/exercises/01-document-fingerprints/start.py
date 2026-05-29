"""
Exercise 01 — Document Fingerprints
The lab needs a count-based fingerprint card for every witness statement in a case.
"""

import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer


def load_statements(path: Path) -> list[dict]:
    """Read witness statements from a JSON file."""
    # TODO: load JSON and return list of statement dicts
    pass


def texts_for_case(statements: list[dict], case_id: str) -> tuple[list[str], list[str]]:
    """Return (doc_ids, texts) for statements matching case_id."""
    # TODO: filter by case_id, return parallel lists of ids and raw_text
    pass


def build_bow_matrix(texts: list[str]):
    """Build a Bag-of-Words matrix. Return (matrix, feature_names)."""
    # TODO: CountVectorizer(stop_words="english"), fit_transform
    pass


def top_terms(matrix, feature_names, doc_index: int, n: int = 5) -> list[tuple[str, int]]:
    """Return the top n terms by raw count for one document."""
    # TODO: sort row by count descending, return (term, count) pairs
    pass


def fingerprint_report(statements: list[dict], case_id: str) -> list[dict]:
    """
    Return a fingerprint card for every statement in a case:
    [{"id", "witness", "top_terms"}, ...] sorted by id.

    top_terms is a list of (term, count) tuples.
    """
    # TODO: build matrix for case texts, top_terms per document
    pass


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"


def main() -> None:
    """Print fingerprint cards for a case."""
    case_id = sys.argv[1] if len(sys.argv) > 1 else "CASE-42"
    statements = load_statements(DATA_PATH)
    report = fingerprint_report(statements, case_id)

    print(f"Inkwell Investigations — Fingerprints for {case_id}")
    print("=" * 60)
    if not report:
        print("No statements found for that case.")
        return

    for card in report:
        terms = ", ".join(f"{t}({c})" for t, c in card["top_terms"])
        print(f"\n{card['id']} — {card['witness']}")
        print(f"  {terms}")


if __name__ == "__main__":
    main()
