"""
Capstone Exercise 03 - NLP Audit Dashboard
Run every NLP technique from the course against a corpus and produce a report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_INKWELL = REPO_ROOT / "data" / "inkwell"
DATA_PUBLIC = REPO_ROOT / "data" / "public"

_nlp = None
_sentiment_pipeline = None


def load_corpus(
    dataset: str = "inkwell",
    path: Path | None = None,
) -> list[dict]:
    """Load records as {id, text, label} dicts.

    Inkwell uses case_id as label. Other datasets use their native labels.
    """
    # TODO
    raise NotImplementedError


def corpus_stats(records: list[dict]) -> dict:
    """Compute corpus-level statistics.

    Return {count, avg_length, min_length, max_length, vocab_size, label_balance}.
    """
    # TODO
    raise NotImplementedError


def discover_topics(texts: list[str], n_topics: int = 3, n_terms: int = 5) -> list[dict]:
    """Run NMF on TF-IDF features.

    Return list of {topic_id, top_terms} dicts.
    """
    # TODO
    raise NotImplementedError


def sentiment_scan(texts: list[str]) -> dict:
    """Run HF sentiment-analysis pipeline on texts.

    Return {distribution: {label: count}, scores: [float]}.
    """
    # TODO
    raise NotImplementedError


def entity_census(texts: list[str]) -> dict[str, dict]:
    """Run spaCy NER across all texts.

    Return {label: {entities: [str], count: int}}.
    Map GPE/FAC to LOC.
    """
    # TODO
    raise NotImplementedError


def classification_probe(records: list[dict]) -> dict | None:
    """If records have labels, train TF-IDF + LR baseline and evaluate.

    Return {accuracy, f1_macro, labels} or None if no labels.
    """
    # TODO
    raise NotImplementedError


def build_report(
    dataset: str,
    stats: dict,
    topics: list[dict],
    sentiment: dict | None,
    entities: dict,
    classification: dict | None,
) -> dict:
    """Assemble a complete JSON-serialisable audit report."""
    # TODO
    raise NotImplementedError


def print_report(report: dict) -> None:
    """Print a formatted CLI summary of the audit report."""
    # TODO
    raise NotImplementedError


def save_report(report: dict, path: Path) -> None:
    """Write the report to a JSON file."""
    # TODO
    raise NotImplementedError


def run_audit(dataset: str, path: Path | None, n_topics: int) -> None:
    """CLI: run full audit and print report."""
    # TODO
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="NLP Audit Dashboard")
    parser.add_argument(
        "command",
        choices=["audit"],
        nargs="?",
        default="audit",
    )
    parser.add_argument("--dataset", default="inkwell")
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--topics", type=int, default=3)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.command == "audit":
        run_audit(args.dataset, args.path, args.topics)


if __name__ == "__main__":
    main()
