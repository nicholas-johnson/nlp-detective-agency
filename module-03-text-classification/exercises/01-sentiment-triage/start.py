"""
Exercise 01 — Sentiment Triage
Flag whether witness statements read calm or hostile before interview.
"""

import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

RANDOM_STATE = 42


def load_sentiment_data(path: Path) -> list[dict]:
    """Read labeled witness sentiment records from JSON."""
    # TODO: load and return list of dicts
    pass


def split_data(records: list[dict]) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return X_train, X_test, y_train, y_test with stratified split."""
    # TODO: extract texts and sentiment labels, train_test_split with stratify
    pass


def build_sentiment_pipeline() -> Pipeline:
    """Return a Pipeline of TfidfVectorizer + MultinomialNB."""
    # TODO: build and return pipeline
    pass


def train_and_evaluate(records: list[dict]) -> dict:
    """
    Train on train split, evaluate on test split.
    Return {"accuracy": float, "f1": float, "predictions": list[dict]}.

    Each prediction dict: {"id", "witness", "actual", "predicted"}.
    """
    # TODO: split, fit pipeline, predict, compute metrics
    pass


def triage_report(records: list[dict]) -> dict:
    """Run full triage workflow and return train_and_evaluate result."""
    # TODO: delegate to train_and_evaluate
    pass


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "witness_sentiment.json"


def main() -> None:
    """Print sentiment triage metrics and misclassifications."""
    records = load_sentiment_data(DATA_PATH)
    report = triage_report(records)

    print("Inkwell Investigations — Sentiment Triage")
    print("=" * 44)
    print(f"Accuracy: {report['accuracy']:.3f}")
    print(f"F1 (hostile): {report['f1']:.3f}")

    errors = [p for p in report["predictions"] if p["actual"] != p["predicted"]]
    if errors:
        print(f"\nMisclassified ({len(errors)}):")
        for p in errors:
            print(f"  {p['id']} {p['witness']}: actual={p['actual']}, predicted={p['predicted']}")
    else:
        print("\nNo misclassifications on test set.")


if __name__ == "__main__":
    main()
