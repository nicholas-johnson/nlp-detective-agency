"""
Exercise 01 - Sentiment Triage (solution)
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
    return json.loads(path.read_text())


def split_data(records: list[dict]) -> tuple[list[str], list[str], list[str], list[str]]:
    texts = [r["text"] for r in records]
    labels = [r["sentiment"] for r in records]
    return train_test_split(
        texts, labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def build_sentiment_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", MultinomialNB()),
    ])


def train_and_evaluate(records: list[dict]) -> dict:
    texts = [r["text"] for r in records]
    labels = [r["sentiment"] for r in records]
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

    pipeline = build_sentiment_pipeline()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    predictions = [
        {
            "id": records[i]["id"],
            "witness": records[i]["witness"],
            "actual": labels[i],
            "predicted": pred,
        }
        for i, pred in zip(test_idx, preds, strict=True)
    ]

    return {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, pos_label="hostile"),
        "predictions": predictions,
    }


def triage_report(records: list[dict]) -> dict:
    return train_and_evaluate(records)


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "witness_sentiment.json"


def main() -> None:
    records = load_sentiment_data(DATA_PATH)
    report = triage_report(records)

    print("Inkwell Investigations - Sentiment Triage")
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
