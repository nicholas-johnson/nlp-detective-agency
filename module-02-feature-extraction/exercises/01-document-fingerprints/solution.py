"""
Exercise 01 — Document Fingerprints (solution)
"""

import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer


def load_statements(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def texts_for_case(statements: list[dict], case_id: str) -> tuple[list[str], list[str]]:
    matched = [s for s in statements if s["case_id"] == case_id]
    return [s["id"] for s in matched], [s["raw_text"] for s in matched]


def build_bow_matrix(texts: list[str]):
    vectorizer = CountVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer.get_feature_names_out()


def top_terms(matrix, feature_names, doc_index: int, n: int = 5) -> list[tuple[str, int]]:
    row = matrix[doc_index].toarray().flatten()
    ranked = row.argsort()[::-1]
    return [
        (feature_names[i], int(row[i]))
        for i in ranked
        if row[i] > 0
    ][:n]


def fingerprint_report(statements: list[dict], case_id: str) -> list[dict]:
    doc_ids, texts = texts_for_case(statements, case_id)
    if not texts:
        return []

    matrix, feature_names = build_bow_matrix(texts)
    report = []
    matched = [s for s in statements if s["case_id"] == case_id]
    for i, stmt in enumerate(matched):
        report.append({
            "id": stmt["id"],
            "witness": stmt["witness"],
            "top_terms": top_terms(matrix, feature_names, i),
        })
    return sorted(report, key=lambda r: r["id"])


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"


def main() -> None:
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
