"""
Exercise 02 - Hoax Filter (solution)
"""

import json
from pathlib import Path

from sklearn.metrics import confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
CLASSIFIER_NAMES = {
    "nb": "Naive Bayes",
    "lr": "Logistic Regression",
    "svm": "Linear SVM",
}


def load_tips(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def build_pipeline(classifier_name: str) -> Pipeline:
    classifiers = {
        "nb": MultinomialNB(),
        "lr": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "svm": LinearSVC(dual="auto", random_state=RANDOM_STATE),
    }
    if classifier_name not in classifiers:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", classifiers[classifier_name]),
    ])


def compare_classifiers(records: list[dict]) -> list[dict]:
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    hoax_f1 = make_scorer(f1_score, pos_label="hoax")
    results = []

    for key in ("nb", "lr", "svm"):
        pipeline = build_pipeline(key)
        scores = cross_val_score(pipeline, texts, labels, cv=5, scoring=hoax_f1)
        results.append({
            "name": CLASSIFIER_NAMES[key],
            "key": key,
            "f1_mean": float(scores.mean()),
            "f1_std": float(scores.std()),
        })

    return sorted(results, key=lambda r: r["f1_mean"], reverse=True)


def confusion(labels_true: list[str], labels_pred: list[str]) -> list[list[int]]:
    cm = confusion_matrix(labels_true, labels_pred, labels=["credible", "hoax"])
    return cm.tolist()


def false_negatives(
    records: list[dict],
    pipeline: Pipeline,
    X_test: list[str],
    y_test: list[str],
) -> list[str]:
    preds = pipeline.predict(X_test)
    id_by_text = {r["text"]: r["id"] for r in records}
    slipped = []
    for text, actual, pred in zip(X_test, y_test, preds, strict=True):
        if actual == "hoax" and pred == "credible":
            slipped.append(id_by_text[text])
    return sorted(slipped)


def hoax_report(records: list[dict]) -> dict:
    comparison = compare_classifiers(records)
    best_key = comparison[0]["key"]

    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    indices = list(range(len(records)))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    X_train = [texts[i] for i in train_idx]
    X_test = [texts[i] for i in test_idx]
    y_train = [labels[i] for i in train_idx]
    y_test = [labels[i] for i in test_idx]

    pipeline = build_pipeline(best_key)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    return {
        "comparison": comparison,
        "best_classifier": comparison[0]["name"],
        "confusion_matrix": confusion(y_test, preds),
        "slipped_through": false_negatives(records, pipeline, X_test, y_test),
    }


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "tips.json"


def main() -> None:
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
