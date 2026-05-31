"""
Exercise 01 - Archive Themes
Run LDA on the cold-case archive and map each file to its dominant theme.
"""

import json
from pathlib import Path

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

RANDOM_STATE = 42


def load_archive(path: Path) -> list[dict]:
    """Read cold-case archive records from JSON."""
    # TODO: load and return list of dicts
    pass


def build_dtm(texts: list[str]):
    """Return document-term matrix and feature names from CountVectorizer."""
    # TODO: CountVectorizer(stop_words="english", min_df=2)
    pass


def fit_lda(dtm, n_topics: int = 4) -> LatentDirichletAllocation:
    """Fit LDA with random_state=42."""
    # TODO: LatentDirichletAllocation and fit
    pass


def top_words(model, feature_names, n: int = 8) -> list[dict]:
    """Return [{topic_id, words: [(term, weight), ...]}, ...] for each topic."""
    # TODO: sort components_ per topic
    pass


def dominant_topics(model, dtm) -> list[tuple[int, float]]:
    """Return (topic_index, weight) for each document's dominant topic."""
    # TODO: transform dtm, argmax per row
    pass


def archive_report(records: list[dict], n_topics: int = 4) -> dict:
    """
    Return {
        "topics": list of topic dicts from top_words,
        "cases": [{id, case_id, title, dominant_topic, weight}, ...],
    }
    """
    # TODO: full workflow
    pass


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "cold_cases.json"


def main() -> None:
    """Print LDA topics and per-case dominant theme assignments."""
    records = load_archive(DATA_PATH)
    report = archive_report(records)

    print("Inkwell Investigations - Archive Themes (LDA)")
    print("=" * 44)

    for topic in report["topics"]:
        words = ", ".join(term for term, _ in topic["words"][:6])
        print(f"\nTopic {topic['topic_id']}: {words}")

    print("\nCase assignments:")
    for case in report["cases"]:
        print(
            f"  {case['id']} ({case['case_id']}) - "
            f"topic {case['dominant_topic']} ({case['weight']:.3f})"
        )


if __name__ == "__main__":
    main()
