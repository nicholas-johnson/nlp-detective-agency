"""
Exercise 02 — Matching Prints (solution)
"""

import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_statements(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def texts_for_case(statements: list[dict], case_id: str) -> tuple[list[str], list[str]]:
    matched = [s for s in statements if s["case_id"] == case_id]
    return [s["id"] for s in matched], [s["raw_text"] for s in matched]


def build_tfidf_matrix(texts: list[str]):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer.get_feature_names_out()


def distinctive_terms(matrix, feature_names, doc_index: int, n: int = 5) -> list[tuple[str, float]]:
    row = matrix[doc_index].toarray().flatten()
    ranked = row.argsort()[::-1]
    return [
        (feature_names[i], round(float(row[i]), 3))
        for i in ranked
        if row[i] > 0
    ][:n]


def most_similar_pair(matrix, doc_ids: list[str]) -> tuple[str, str, float] | None:
    if len(doc_ids) < 2:
        return None

    sims = cosine_similarity(matrix)
    best_score = -1.0
    best_pair = None

    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            score = float(sims[i][j])
            if score > best_score:
                best_score = score
                best_pair = (doc_ids[i], doc_ids[j], score)

    return best_pair


def compare_ngram_vocab_sizes(texts: list[str]) -> dict[str, int]:
    unigram_vec = CountVectorizer(stop_words="english", ngram_range=(1, 1))
    bigram_vec = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    unigram_vec.fit(texts)
    bigram_vec.fit(texts)
    return {
        "unigram": len(unigram_vec.get_feature_names_out()),
        "bigram": len(bigram_vec.get_feature_names_out()),
    }


def similarity_report(statements: list[dict], case_id: str) -> dict:
    doc_ids, texts = texts_for_case(statements, case_id)
    if not texts:
        return {
            "case_id": case_id,
            "most_similar": None,
            "ngram_vocab_sizes": {"unigram": 0, "bigram": 0},
        }

    matrix, _ = build_tfidf_matrix(texts)
    return {
        "case_id": case_id,
        "most_similar": most_similar_pair(matrix, doc_ids),
        "ngram_vocab_sizes": compare_ngram_vocab_sizes(texts),
    }


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"


def main() -> None:
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
