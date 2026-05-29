"""
Exercise 02 — Matching Prints
Two witnesses may be telling the same story. TF-IDF finds the closest pair.
"""

import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def load_statements(path: Path) -> list[dict]:
    """Read witness statements from a JSON file."""
    # TODO: load JSON and return list of statement dicts
    pass


def texts_for_case(statements: list[dict], case_id: str) -> tuple[list[str], list[str]]:
    """Return (doc_ids, texts) for statements matching case_id."""
    # TODO: filter by case_id
    pass


def build_tfidf_matrix(texts: list[str]):
    """Build a TF-IDF matrix. Return (matrix, feature_names)."""
    # TODO: TfidfVectorizer(stop_words="english"), fit_transform
    pass


def distinctive_terms(matrix, feature_names, doc_index: int, n: int = 5) -> list[tuple[str, float]]:
    """Return the top n terms by TF-IDF weight for one document."""
    # TODO: sort row by weight descending
    pass


def most_similar_pair(matrix, doc_ids: list[str]) -> tuple[str, str, float] | None:
    """Return (id_a, id_b, score) for the highest off-diagonal cosine similarity."""
    # TODO: cosine_similarity, exclude diagonal, return best pair
    pass


def compare_ngram_vocab_sizes(texts: list[str]) -> dict[str, int]:
    """Return vocabulary sizes for unigram (1,1) and bigram (1,2) ranges."""
    # TODO: CountVectorizer with ngram_range (1,1) and (1,2)
    pass


def similarity_report(statements: list[dict], case_id: str) -> dict:
    """
    Return a report dict:
    {
        "case_id": str,
        "most_similar": (id_a, id_b, score) or None,
        "ngram_vocab_sizes": {"unigram": int, "bigram": int},
    }
    """
    # TODO: combine most_similar_pair and compare_ngram_vocab_sizes
    pass


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"


def main() -> None:
    """Print a similarity report for a case."""
    case_id = sys.argv[1] if len(sys.argv) > 1 else "CASE-42"
    statements = load_statements(DATA_PATH)
    report = similarity_report(statements, case_id)

    print(f"Inkwell Investigations — Similarity Report for {case_id}")
    print("=" * 60)
    if report["most_similar"] is None:
        print("No statements found for that case.")
        return

    id_a, id_b, score = report["most_similar"]
    print(f"\nMost similar pair: {id_a} <-> {id_b}  (score: {score:.3f})")

    sizes = report["ngram_vocab_sizes"]
    print(f"\nVocabulary sizes:")
    print(f"  Unigrams only:  {sizes['unigram']}")
    print(f"  With bigrams:   {sizes['bigram']}")


if __name__ == "__main__":
    main()
