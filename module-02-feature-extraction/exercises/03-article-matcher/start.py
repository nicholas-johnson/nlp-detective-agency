"""
Exercise 03 - Article Matcher
Apply TF-IDF similarity to real newsgroup articles and check whether
the most similar pairs share the same category.
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_articles(path: Path) -> list[dict]:
    """Read newsgroup article records from JSON."""
    # TODO: load and return list of dicts
    pass


def build_tfidf_matrix(texts: list[str]):
    """Return TF-IDF matrix and feature names."""
    # TODO: TfidfVectorizer(stop_words="english", min_df=2)
    pass


def similarity_matrix(tfidf_matrix):
    """Return cosine similarity matrix."""
    # TODO: cosine_similarity
    pass


def top_similar_pairs(sim_matrix, ids: list[str], n: int = 5) -> list[dict]:
    """Return top n pairs sorted by score: {id_a, id_b, score}."""
    # TODO: iterate upper triangle, sort descending
    pass


def same_category_rate(pairs: list[dict], records: list[dict]) -> float:
    """Fraction of pairs where both articles share the same category."""
    # TODO
    pass


def compare_ngram_vocab_sizes(texts: list[str]) -> dict[str, int]:
    """Return {unigram: int, bigram: int} vocabulary sizes."""
    # TODO
    pass


def article_audit(records: list[dict], n_pairs: int = 5) -> dict:
    """
    Return {
        "total_articles": int,
        "vocab_sizes": {unigram, bigram},
        "top_pairs": [...],
        "same_category_rate": float,
    }
    """
    # TODO: full workflow
    pass


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "newsgroups_articles.json"


def main() -> None:
    records = load_articles(DATA_PATH)
    report = article_audit(records)

    print("20 Newsgroups - Similarity Audit")
    print("=" * 40)
    print(f"Articles: {report['total_articles']}")
    print(f"Same-category rate (top pairs): {report['same_category_rate']:.0%}")

    sizes = report["vocab_sizes"]
    print(f"\nVocabulary: unigram={sizes['unigram']}, bigram={sizes['bigram']}")

    print("\nTop similar pairs:")
    for pair in report["top_pairs"]:
        print(f"  {pair['id_a']} <-> {pair['id_b']}  ({pair['score']:.3f})")


if __name__ == "__main__":
    main()
