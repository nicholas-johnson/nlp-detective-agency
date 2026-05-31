"""
Exercise 03 - Real-World Topics
Apply topic modelling to a real public dataset (20 Newsgroups) and
evaluate whether discovered topics align with actual categories.
"""

import json
from pathlib import Path

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RANDOM_STATE = 42


def load_articles(path: Path) -> list[dict]:
    """Read newsgroup articles from JSON."""
    # TODO: load and return list of dicts
    pass


def build_dtm(texts: list[str]):
    """Return document-term matrix and feature names from CountVectorizer."""
    # TODO: CountVectorizer(stop_words="english", min_df=2)
    pass


def build_tfidf(texts: list[str]):
    """Return TF-IDF matrix and feature names."""
    # TODO: TfidfVectorizer(stop_words="english", min_df=2)
    pass


def fit_lda(dtm, n_topics: int = 4) -> LatentDirichletAllocation:
    """Fit LDA with random_state=42."""
    # TODO: LatentDirichletAllocation and fit
    pass


def top_words(model, feature_names, n: int = 8) -> list[list[str]]:
    """Return list of word lists, one per topic."""
    # TODO: sort components_ per topic
    pass


def dominant_topic(model, dtm) -> list[int]:
    """Return dominant topic index per document."""
    # TODO: transform, argmax per row
    pass


def topic_purity(assignments: list[int], true_labels: list[str]) -> list[dict]:
    """
    For each discovered topic, compute the purity:
    majority_count / total_in_topic.

    Return [{topic_id, majority_category, count, total, purity}, ...]
    """
    # TODO: group by topic, find majority category, compute purity
    pass


def contingency_matrix(assignments: list[int], true_labels: list[str], n_topics: int) -> dict:
    """
    Return {categories: sorted unique labels,
            matrix: [[count, ...], ...]}
    where rows = topic ids, columns = categories.
    """
    # TODO: build n_topics x n_categories count matrix
    pass


def topic_audit(records: list[dict], n_topics: int = 4) -> dict:
    """
    Full audit pipeline. Return {
        "n_topics": int,
        "topics": [[word, ...], ...],
        "purity": [{"topic_id", "majority_category", "purity"}, ...],
        "avg_purity": float,
        "contingency": {"categories": [...], "matrix": [[...], ...]},
        "assignments": [{"id", "category", "topic"}, ...],
    }
    """
    # TODO: build DTM, fit LDA, compute purity and contingency
    pass


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "newsgroups_sample.json"


def main() -> None:
    """Print topic audit results."""
    records = load_articles(DATA_PATH)
    report = topic_audit(records)

    print("Topic Modelling - 20 Newsgroups Audit")
    print("=" * 44)

    for i, words in enumerate(report["topics"]):
        print(f"\nTopic {i}: {', '.join(words[:6])}")

    print(f"\nPurity per topic:")
    for p in report["purity"]:
        print(f"  Topic {p['topic_id']}: {p['majority_category']} - {p['purity']:.0%} ({p['count']}/{p['total']})")

    print(f"\nAverage purity: {report['avg_purity']:.0%}")

    cm = report["contingency"]
    cats = cm["categories"]
    header = "          " + "  ".join(f"{c[:8]:>8}" for c in cats)
    print(f"\nContingency matrix (topic x category):\n{header}")
    for i, row in enumerate(cm["matrix"]):
        vals = "  ".join(f"{v:>8}" for v in row)
        print(f"  Topic {i}  {vals}")


if __name__ == "__main__":
    main()
