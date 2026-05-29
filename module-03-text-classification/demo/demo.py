"""
Inkwell Investigations — Triage Desk (demo)
Run:  python module-03-text-classification/demo/demo.py
"""

import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix, f1_score, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "inkwell"
SENTIMENT_PATH = DATA_DIR / "witness_sentiment.json"
TIPS_PATH = DATA_DIR / "tips.json"
RANDOM_STATE = 42


def load_sentiment() -> list[dict]:
    return json.loads(SENTIMENT_PATH.read_text())


def load_tips() -> list[dict]:
    return json.loads(TIPS_PATH.read_text())


def build_sentiment_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", MultinomialNB()),
    ])


def build_hoax_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", MultinomialNB()),
    ])


def build_pipeline(name: str) -> Pipeline:
    classifiers = {
        "nb": MultinomialNB(),
        "lr": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "svm": LinearSVC(dual="auto", random_state=RANDOM_STATE),
    }
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", classifiers[name]),
    ])


def train_test(records: list[dict], label_key: str) -> tuple:
    texts = [r["text"] for r in records]
    labels = [r[label_key] for r in records]
    return train_test_split(
        texts, labels,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def pick_record(records: list[dict], label: str = "record") -> dict | None:
    for i, r in enumerate(records, 1):
        extra = r.get("witness", r.get("id", ""))
        print(f"  {i}. {r['id']} — {extra}")
    choice = input(f"Pick a {label}: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(records)):
        print("Invalid choice.")
        return None
    return records[int(choice) - 1]


def main() -> None:
    print("Inkwell Investigations — Triage Desk")
    print("=" * 40)

    while True:
        print("\n1. List datasets")
        print("2. Sentiment — predict one witness statement")
        print("3. Hoax — predict one tip")
        print("4. Classifier shootout on tips (NB / LR / SVM)")
        print("5. Classification report for hoax detector")
        print("0. Quit")
        choice = input("\nChoice: ").strip()

        if choice == "0":
            break

        if choice == "1":
            sentiment = load_sentiment()
            tips = load_tips()
            print("\nWitness sentiment:")
            for label, count in sorted(Counter(r["sentiment"] for r in sentiment).items()):
                print(f"  {label}: {count}")
            print("\nTips:")
            for label, count in sorted(Counter(r["label"] for r in tips).items()):
                print(f"  {label}: {count}")
            continue

        if choice == "2":
            records = load_sentiment()
            X_train, X_test, y_train, y_test = train_test(records, "sentiment")
            pipeline = build_sentiment_pipeline()
            pipeline.fit(X_train, y_train)
            record = pick_record(records, "statement")
            if record is None:
                continue
            pred = pipeline.predict([record["text"]])[0]
            print(f"\nPrediction: {pred}  (actual: {record['sentiment']})")

        elif choice == "3":
            records = load_tips()
            X_train, X_test, y_train, y_test = train_test(records, "label")
            pipeline = build_hoax_pipeline()
            pipeline.fit(X_train, y_train)
            record = pick_record(records, "tip")
            if record is None:
                continue
            pred = pipeline.predict([record["text"]])[0]
            print(f"\nPrediction: {pred}  (actual: {record['label']})")

        elif choice == "4":
            records = load_tips()
            texts = [r["text"] for r in records]
            labels = [r["label"] for r in records]
            print(f"\n{'Classifier':<20} {'F1 mean':>8} {'F1 std':>8}")
            print("-" * 40)
            hoax_f1 = make_scorer(f1_score, pos_label="hoax")
            for name, label in [("nb", "Naive Bayes"), ("lr", "Logistic Reg"), ("svm", "Linear SVM")]:
                pipeline = build_pipeline(name)
                scores = cross_val_score(pipeline, texts, labels, cv=5, scoring=hoax_f1)
                print(f"{label:<20} {scores.mean():>8.3f} {scores.std():>8.3f}")

        elif choice == "5":
            records = load_tips()
            X_train, X_test, y_train, y_test = train_test(records, "label")
            pipeline = build_hoax_pipeline()
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            print("\nClassification report:")
            print(classification_report(y_test, preds, labels=["credible", "hoax"]))
            print("Confusion matrix (rows=actual, cols=predicted):")
            cm = confusion_matrix(y_test, preds, labels=["credible", "hoax"])
            print(f"              credible  hoax")
            for i, label in enumerate(["credible", "hoax"]):
                print(f"  {label:<10}  {cm[i][0]:>8}  {cm[i][1]:>4}")

        else:
            print("Unknown option.")


if __name__ == "__main__":
    main()
