"""
Inkwell Investigations - Triage Desk (demo)
Run:  python module-03-text-classification/demo/demo.py
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
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
        print(f"  {i}. {r['id']} - {extra}")
    choice = input(f"Pick a {label}: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(records)):
        print("Invalid choice.")
        return None
    return records[int(choice) - 1]


def _tip_feature_vector(pipeline: Pipeline, text: str):
    """Return the TF-IDF vector, feature names, and non-zero indices for one document."""
    tfidf = pipeline.named_steps["tfidf"]
    vec = tfidf.transform([text])
    features = tfidf.get_feature_names_out()
    nonzero = vec.nonzero()[1]
    return vec, features, nonzero


def predict_tip(classifier_key: str, records: list[dict]) -> None:
    X_train, X_test, y_train, y_test = train_test(records, "label")
    pipeline = build_pipeline(classifier_key)
    pipeline.fit(X_train, y_train)

    record = pick_record(records, "tip")
    if record is None:
        return

    text = record["text"]
    pred = pipeline.predict([text])[0]
    print(f"\nPrediction: {pred}  (actual: {record['label']})")

    vec, features, nonzero = _tip_feature_vector(pipeline, text)
    clf = pipeline.named_steps["clf"]

    if classifier_key == "nb":
        proba = pipeline.predict_proba([text])[0]
        classes = list(pipeline.classes_)
        print("\nPosterior probabilities:")
        for cls, p in zip(classes, proba):
            marker = " ←" if cls == pred else ""
            print(f"  {cls}: {p:.3f}{marker}")

        log_prob = clf.feature_log_prob_
        print("\nTop words from this tip (by log-likelihood difference):")
        diffs = {}
        hoax_idx = classes.index("hoax")
        cred_idx = classes.index("credible")
        for j in nonzero:
            diff = log_prob[hoax_idx, j] - log_prob[cred_idx, j]
            diffs[j] = diff
        ranked = sorted(diffs, key=lambda j: abs(diffs[j]), reverse=True)[:8]
        for j in ranked:
            direction = "hoax" if diffs[j] > 0 else "credible"
            print(f"  {features[j]:<20} log-diff {diffs[j]:+.2f}  → {direction}")

    elif classifier_key == "lr":
        proba = clf.predict_proba([vec.toarray()[0]])[0]
        classes = list(clf.classes_)
        print("\nClass probabilities:")
        for cls, p in zip(classes, proba):
            marker = " ←" if cls == pred else ""
            print(f"  {cls}: {p:.3f}{marker}")

        weights = clf.coef_[0]
        tfidf_vals = vec.toarray()[0]
        contributions = weights * tfidf_vals

        print("\nWords from this tip (by contribution to score):")
        active = [(j, contributions[j]) for j in nonzero if abs(contributions[j]) > 1e-6]
        active.sort(key=lambda x: abs(x[1]), reverse=True)
        hoax_idx = classes.index("hoax")
        sign = 1 if hoax_idx == 1 else -1
        for j, contrib in active[:8]:
            direction = "hoax" if contrib * sign > 0 else "credible"
            print(f"  {features[j]:<20} weight {weights[j]:+.3f}  tfidf {tfidf_vals[j]:.2f}  contrib {contrib:+.3f}  → {direction}")

        print("\nTop model features overall (hoax):")
        top_hoax = np.argsort(weights * sign)[-5:][::-1]
        for i in top_hoax:
            print(f"  {features[i]:<20} {weights[i]:+.3f}")
        print("Top model features overall (credible):")
        top_credible = np.argsort(weights * sign)[:5]
        for i in top_credible:
            print(f"  {features[i]:<20} {weights[i]:+.3f}")

    elif classifier_key == "svm":
        weights = clf.coef_[0]
        classes = list(clf.classes_)
        hoax_idx = classes.index("hoax")
        sign = 1 if hoax_idx == 1 else -1
        tfidf_vals = vec.toarray()[0]
        contributions = weights * tfidf_vals
        decision = clf.decision_function(vec)[0]

        print(f"\nDecision score: {decision:+.3f}  ({'hoax side' if decision * sign > 0 else 'credible side'})")

        print("\nWords from this tip (by contribution to decision):")
        active = [(j, contributions[j]) for j in nonzero if abs(contributions[j]) > 1e-6]
        active.sort(key=lambda x: abs(x[1]), reverse=True)
        for j, contrib in active[:8]:
            direction = "hoax" if contrib * sign > 0 else "credible"
            print(f"  {features[j]:<20} weight {weights[j]:+.3f}  tfidf {tfidf_vals[j]:.2f}  contrib {contrib:+.3f}  → {direction}")

        preds = pipeline.predict(X_test)
        f1 = f1_score(y_test, preds, pos_label="hoax")
        print(f"\nHold-out F1 (hoax): {f1:.3f}")


def main() -> None:
    print("Inkwell Investigations - Triage Desk")
    print("=" * 40)

    while True:
        print("\n1. List datasets")
        print("2. Sentiment pipeline - predict one witness statement")
        print("3. Naive Bayes - predict one tip")
        print("4. Logistic Regression - predict one tip + top features")
        print("5. Linear SVM - predict one tip + hold-out F1")
        print("6. Classifier shootout on tips (NB / LR / SVM)")
        print("7. Classification report + confusion matrix")
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

        elif choice in ("3", "4", "5"):
            key = {"3": "nb", "4": "lr", "5": "svm"}[choice]
            predict_tip(key, load_tips())

        elif choice == "6":
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

        elif choice == "7":
            records = load_tips()
            X_train, X_test, y_train, y_test = train_test(records, "label")
            pipeline = build_pipeline("nb")
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
