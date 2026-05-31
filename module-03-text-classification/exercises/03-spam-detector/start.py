"""
Exercise 03 - Spam Detector
Apply the hoax-filter classifier shootout to real SMS spam data.
"""

import json
from pathlib import Path

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

RANDOM_STATE = 42


def load_messages(path: Path) -> list[dict]:
    """Read SMS message records from JSON."""
    # TODO: load and return list of dicts
    pass


def build_pipeline(classifier_name: str) -> Pipeline:
    """Build Pipeline with TfidfVectorizer and the named classifier."""
    # TODO: "nb" -> MultinomialNB, "lr" -> LogisticRegression, "svm" -> LinearSVC
    pass


def compare_classifiers(records: list[dict]) -> list[dict]:
    """Return sorted list of {name, f1_mean, f1_std} from 5-fold cross-validation."""
    # TODO: cross_val_score with make_scorer(f1_score, pos_label="spam")
    pass


def confusion(labels_true: list[str], labels_pred: list[str]) -> list[list[int]]:
    """Return 2x2 confusion matrix [ham][spam] as nested lists."""
    # TODO: confusion_matrix with labels=["ham", "spam"]
    pass


def false_negatives(
    records: list[dict],
    pipeline: Pipeline,
    X_test: list[str],
    y_test: list[str],
) -> list[str]:
    """Return ids of spam messages predicted as ham."""
    # TODO
    pass


def spam_report(records: list[dict]) -> dict:
    """
    Return {
        "comparison": list of classifier results,
        "best_classifier": str,
        "confusion_matrix": [[...], [...]],
        "slipped_through": list of message ids,
    }
    """
    # TODO: compare classifiers, train best on train split, evaluate on test
    pass


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "sms_spam_sample.json"


def main() -> None:
    records = load_messages(DATA_PATH)
    report = spam_report(records)

    print("SMS Spam - Classifier Report")
    print("=" * 40)

    print(f"\n{'Classifier':<20} {'F1 mean':>8} {'F1 std':>8}")
    print("-" * 40)
    for row in report["comparison"]:
        print(f"{row['name']:<20} {row['f1_mean']:>8.3f} {row['f1_std']:>8.3f}")

    print(f"\nBest classifier: {report['best_classifier']}")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    cm = report["confusion_matrix"]
    print(f"              ham  spam")
    print(f"  ham         {cm[0][0]:>4}  {cm[0][1]:>4}")
    print(f"  spam        {cm[1][0]:>4}  {cm[1][1]:>4}")

    slipped = report["slipped_through"]
    if slipped:
        print(f"\nSpam that slipped through ({len(slipped)}): {', '.join(slipped)}")
    else:
        print("\nNo spam slipped through on test set.")


if __name__ == "__main__":
    main()
