"""
Exercise 02 - Hoax Filter
Compare classifiers on tip credibility and find hoaxes that slip through.
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


def load_tips(path: Path) -> list[dict]:
    """Read labeled tip records from JSON."""
    # TODO: load and return list of dicts
    pass


def build_pipeline(classifier_name: str) -> Pipeline:
    """Build Pipeline with TfidfVectorizer and the named classifier."""
    # TODO: "nb" -> MultinomialNB, "lr" -> LogisticRegression, "svm" -> LinearSVC
    pass


def compare_classifiers(records: list[dict]) -> list[dict]:
    """Return sorted list of {name, f1_mean, f1_std} from 5-fold cross-validation."""
    # TODO: cross_val_score with make_scorer(f1_score, pos_label="hoax") for nb, lr, svm
    pass


def confusion(labels_true: list[str], labels_pred: list[str]) -> list[list[int]]:
    """Return 2x2 confusion matrix [credible][hoax] as nested lists."""
    # TODO: use sklearn.metrics.confusion_matrix with labels=["credible", "hoax"]
    pass


def false_negatives(
    records: list[dict],
    pipeline: Pipeline,
    X_test: list[str],
    y_test: list[str],
) -> list[str]:
    """Return ids of hoax tips predicted as credible."""
    # TODO: predict, find hoax records misclassified as credible
    pass


def hoax_report(records: list[dict]) -> dict:
    """
    Return {
        "comparison": list of classifier results,
        "best_classifier": str,
        "confusion_matrix": [[...], [...]],
        "slipped_through": list of tip ids,
    }
    """
    # TODO: compare classifiers, train best on train split, evaluate on test
    pass


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "tips.json"


def main() -> None:
    """Print classifier shootout and hoaxes that slipped through."""
    records = load_tips(DATA_PATH)
    report = hoax_report(records)

    print("Inkwell Investigations - Hoax Filter Report")
    print("=" * 44)

    print(f"\n{'Classifier':<20} {'F1 mean':>8} {'F1 std':>8}")
    print("-" * 40)
    for row in report["comparison"]:
        print(f"{row['name']:<20} {row['f1_mean']:>8.3f} {row['f1_std']:>8.3f}")

    print(f"\nBest classifier: {report['best_classifier']}")
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    cm = report["confusion_matrix"]
    print(f"              credible  hoax")
    print(f"  credible    {cm[0][0]:>8}  {cm[0][1]:>4}")
    print(f"  hoax        {cm[1][0]:>8}  {cm[1][1]:>4}")

    slipped = report["slipped_through"]
    if slipped:
        print(f"\nHoaxes that slipped through ({len(slipped)}): {', '.join(slipped)}")
    else:
        print("\nNo hoaxes slipped through on test set.")


if __name__ == "__main__":
    main()
