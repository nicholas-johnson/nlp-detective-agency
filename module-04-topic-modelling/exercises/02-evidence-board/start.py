"""
Exercise 02 - Evidence Board
Compare LDA and NMF, pick the best topic count, and produce an evidence-board briefing.
"""

import json
from pathlib import Path

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RANDOM_STATE = 42


def load_archive(path: Path) -> list[dict]:
    """Read cold-case archive records from JSON."""
    # TODO: load and return list of dicts
    pass


def perplexity_scores(dtm, k_values: list[int]) -> list[dict]:
    """Return [{k, perplexity}, ...] sorted by k."""
    # TODO: fit LDA for each k, record perplexity
    pass


def fit_nmf(tfidf_matrix, n_topics: int) -> NMF:
    """Fit NMF with random_state=42."""
    # TODO: NMF and fit
    pass


def compare_models(lda_model, nmf_model, feature_names) -> dict:
    """Return {lda: [...], nmf: [...]} with top-8 words per topic for each model."""
    # TODO: extract top words from both models
    pass


def label_topics(top_words: list[dict]) -> dict[int, str]:
    """Map topic_id to human label using keyword rules."""
    # TODO: keyword rules for Waterfront, Financial, Surveillance, Neighbourhood, General
    pass


def evidence_board(records: list[dict], k_values: list[int] | None = None) -> dict:
    """
    Return {
        "best_k": int,
        "lda_topics": [...],
        "nmf_topics": [...],
        "board": [{topic_label, case_ids}, ...],
    }
    """
    # TODO: pick best k by perplexity, fit LDA+NMF, group cases by dominant topic
    pass


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "cold_cases.json"


def main() -> None:
    """Print best topic count, model comparison, and evidence-board groupings."""
    records = load_archive(DATA_PATH)
    report = evidence_board(records)

    print("Inkwell Investigations - Evidence Board")
    print("=" * 44)
    print(f"\nBest topic count (lowest perplexity): k={report['best_k']}")

    print("\nLDA vs NMF top words:")
    for topic_id, lda_words in enumerate(report["lda_topics"]):
        nmf_words = report["nmf_topics"][topic_id]
        print(f"  Topic {topic_id}")
        print(f"    LDA: {', '.join(lda_words[:6])}")
        print(f"    NMF: {', '.join(nmf_words[:6])}")

    print("\nEvidence board:")
    for group in report["board"]:
        ids = ", ".join(group["case_ids"])
        print(f"  {group['topic_label']}: {ids}")


if __name__ == "__main__":
    main()
