"""
Capstone - Open Your Case (solution)
End-to-end text classification: load real data, baseline, zero-shot, deploy.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
import urllib.request
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_PUBLIC = REPO_ROOT / "data" / "public"
CACHE_DIR = Path.home() / ".cache" / "inkwell" / "datasets"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

RANDOM_STATE = 42
SMS_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
)
NEWGROUPS_CATEGORIES = ["sci.med", "rec.autos", "comp.graphics", "talk.politics.misc"]
AG_NEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
CLASSIFIER_NAMES = {"nb": "Naive Bayes", "lr": "Logistic Regression", "svm": "Linear SVM"}

DATASET_KEYS = ("sms_spam", "newsgroups", "ag_news", "imdb", "movie_reviews", "custom")

_zero_shot_pipeline = None


def list_datasets() -> list[str]:
    return list(DATASET_KEYS)


def _apply_limit(records: list[dict], limit: int | None) -> list[dict]:
    if limit is not None and limit > 0:
        return records[:limit]
    return records


def _load_sms_spam(limit: int | None) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / "SMSSpamCollection"
    if not cache_file.exists():
        print(f"Downloading SMS Spam Collection from UCI...")
        with urllib.request.urlopen(SMS_UCI_URL, timeout=60) as response:
            data = response.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            zf.extract("SMSSpamCollection", CACHE_DIR)

    records = []
    for idx, line in enumerate(cache_file.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        label, text = line.split("\t", maxsplit=1)
        records.append({"id": f"SMS-{idx:05d}", "text": text, "label": label})
    return _apply_limit(records, limit)


def _load_newsgroups(limit: int | None) -> list[dict]:
    from sklearn.datasets import fetch_20newsgroups

    data = fetch_20newsgroups(
        subset="train",
        categories=NEWGROUPS_CATEGORIES,
        remove=("headers", "footers", "quotes"),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    records = [
        {
            "id": f"NEWS-{idx:05d}",
            "text": text,
            "label": NEWGROUPS_CATEGORIES[target],
        }
        for idx, (text, target) in enumerate(zip(data.data, data.target, strict=True), start=1)
    ]
    return _apply_limit(records, limit)


def _load_ag_news(limit: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("ag_news", split="train")
    records = []
    for idx, row in enumerate(ds):
        records.append({
            "id": f"AG-{idx:05d}",
            "text": row["text"],
            "label": AG_NEWS_LABELS[row["label"]],
        })
    return _apply_limit(records, limit)


def _load_imdb(limit: int | None) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("imdb", split="train")
    label_map = {0: "neg", 1: "pos"}
    records = []
    for idx, row in enumerate(ds):
        records.append({
            "id": f"IMDB-{idx:05d}",
            "text": row["text"],
            "label": label_map[row["label"]],
        })
    return _apply_limit(records, limit)


def _load_movie_reviews(limit: int | None) -> list[dict]:
    path = DATA_PUBLIC / "movie_reviews_sample.json"
    raw = json.loads(path.read_text())
    records = [
        {"id": r["id"], "text": r["text"], "label": r["sentiment"]}
        for r in raw
    ]
    return _apply_limit(records, limit)


def _load_custom(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Custom dataset not found: {path}")

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text())
        if not isinstance(raw, list):
            raise ValueError("Custom JSON must be a list of records")
        records = []
        for idx, row in enumerate(raw):
            records.append({
                "id": str(row.get("id", f"CUSTOM-{idx + 1:05d}")),
                "text": str(row["text"]),
                "label": str(row["label"]),
            })
        return records

    if path.suffix.lower() == ".csv":
        records = []
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                records.append({
                    "id": str(row.get("id", f"CUSTOM-{idx + 1:05d}")),
                    "text": str(row["text"]),
                    "label": str(row["label"]),
                })
        return records

    raise ValueError("Custom path must be .json or .csv")


def load_dataset(
    name: str,
    *,
    path: Path | None = None,
    limit: int | None = None,
) -> list[dict]:
    if name not in DATASET_KEYS:
        raise ValueError(f"Unknown dataset: {name}. Choose from {list_datasets()}")

    loaders = {
        "sms_spam": lambda: _load_sms_spam(limit),
        "newsgroups": lambda: _load_newsgroups(limit),
        "ag_news": lambda: _load_ag_news(limit),
        "imdb": lambda: _load_imdb(limit),
        "movie_reviews": lambda: _load_movie_reviews(limit),
    }

    if name == "custom":
        if path is None:
            raise ValueError("custom dataset requires --path")
        return _load_custom(path)

    return loaders[name]()


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def explore_dataset(records: list[dict]) -> dict:
    labels = [r["label"] for r in records]
    lengths = [len(r["text"]) for r in records]
    balance = dict(Counter(labels))
    return {
        "count": len(records),
        "classes": sorted(balance.keys()),
        "class_balance": balance,
        "avg_length": round(statistics.mean(lengths), 1) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "sample": records[0] if records else None,
    }


def split_records(
    records: list[dict],
    test_size: float = 0.25,
) -> tuple[list[dict], list[dict]]:
    if len(records) < 4:
        raise ValueError("Need at least 4 records for train/test split")
    labels = [r["label"] for r in records]
    train, test = train_test_split(
        records,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    return train, test


def build_pipeline(classifier_name: str = "lr") -> Pipeline:
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


def train_baseline(
    train_records: list[dict],
    classifier_name: str = "lr",
) -> Pipeline:
    texts = [preprocess_text(r["text"]) for r in train_records]
    labels = [r["label"] for r in train_records]
    pipeline = build_pipeline(classifier_name)
    pipeline.fit(texts, labels)
    return pipeline


def evaluate(
    model: Pipeline,
    texts: list[str],
    labels: list[str],
) -> dict:
    preds = model.predict(texts)
    labels_sorted = sorted(set(labels))
    report = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(labels, preds)), 3),
        "f1_macro": round(float(f1_score(labels, preds, average="macro")), 3),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(labels, preds, labels=labels_sorted).tolist(),
        "label_order": labels_sorted,
        "predictions": preds.tolist(),
    }


def zero_shot_predict(texts: list[str], labels: list[str]) -> list[dict]:
    global _zero_shot_pipeline
    from transformers import pipeline

    if _zero_shot_pipeline is None:
        _zero_shot_pipeline = pipeline("zero-shot-classification")

    if isinstance(texts, str):
        texts = [texts]

    results = _zero_shot_pipeline(
        texts,
        candidate_labels=labels,
        truncation=True,
        max_length=512,
    )
    if isinstance(results, dict):
        return [results]
    return results


def zero_shot_evaluate(
    test_records: list[dict],
    label_names: list[str],
) -> dict:
    texts = [preprocess_text(r["text"]) for r in test_records]
    true_labels = [r["label"] for r in test_records]
    results = zero_shot_predict(texts, label_names)
    preds = [r["labels"][0] for r in results]
    confidences = [round(float(r["scores"][0]), 3) for r in results]
    labels_sorted = sorted(set(true_labels))
    return {
        "accuracy": round(float(accuracy_score(true_labels, preds)), 3),
        "f1_macro": round(float(f1_score(true_labels, preds, average="macro")), 3),
        "confusion_matrix": confusion_matrix(
            true_labels, preds, labels=labels_sorted
        ).tolist(),
        "label_order": labels_sorted,
        "predictions": preds,
        "confidences": confidences,
    }


def compare_models(
    baseline_metrics: dict,
    transformer_metrics: dict,
) -> dict:
    return {
        "baseline": {
            "accuracy": baseline_metrics["accuracy"],
            "f1_macro": baseline_metrics["f1_macro"],
        },
        "transformer": {
            "accuracy": transformer_metrics["accuracy"],
            "f1_macro": transformer_metrics["f1_macro"],
        },
        "winner": (
            "baseline"
            if baseline_metrics["f1_macro"] >= transformer_metrics["f1_macro"]
            else "transformer"
        ),
    }


def error_analysis(
    test_records: list[dict],
    predictions: list[str],
) -> list[dict]:
    errors = []
    for record, pred in zip(test_records, predictions, strict=True):
        if pred != record["label"]:
            errors.append({
                "id": record["id"],
                "text": record["text"][:200],
                "actual": record["label"],
                "predicted": pred,
            })
    return errors


def save_artifacts(
    pipeline: Pipeline,
    dataset_name: str,
    labels: list[str],
    metrics: dict,
    classifier_name: str,
    artifacts_dir: Path | None = None,
) -> Path:
    out_dir = (artifacts_dir or ARTIFACTS_DIR) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, out_dir / "baseline.joblib")
    config = {
        "dataset": dataset_name,
        "labels": labels,
        "classifier": classifier_name,
        "saved_at": datetime.now(UTC).isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return out_dir


def load_artifacts(dataset_name: str, artifacts_dir: Path | None = None) -> tuple[Pipeline, dict]:
    base = (artifacts_dir or ARTIFACTS_DIR) / dataset_name
    pipeline = joblib.load(base / "baseline.joblib")
    config = json.loads((base / "config.json").read_text())
    return pipeline, config


def run_explore(dataset: str, path: Path | None, limit: int | None) -> None:
    records = load_dataset(dataset, path=path, limit=limit)
    stats = explore_dataset(records)
    print(f"Dataset: {dataset} ({stats['count']} records)")
    print(f"Classes: {', '.join(stats['classes'])}")
    print(f"Balance: {stats['class_balance']}")
    print(f"Length - avg: {stats['avg_length']}, min: {stats['min_length']}, max: {stats['max_length']}")
    if stats["sample"]:
        s = stats["sample"]
        print(f"\nSample [{s['id']}] ({s['label']}): {s['text'][:120]}...")


def run_train(
    dataset: str,
    path: Path | None,
    limit: int | None,
    classifier: str,
) -> None:
    records = load_dataset(dataset, path=path, limit=limit)
    train, test = split_records(records)
    pipeline = train_baseline(train, classifier)
    metrics = evaluate(
        pipeline,
        [preprocess_text(r["text"]) for r in test],
        [r["label"] for r in test],
    )
    labels = sorted({r["label"] for r in records})
    out = save_artifacts(pipeline, dataset, labels, metrics, classifier)
    print(f"Trained {CLASSIFIER_NAMES[classifier]} on {dataset}")
    print(f"Test accuracy: {metrics['accuracy']}, F1 macro: {metrics['f1_macro']}")
    print(f"Artifacts saved to {out}")


def run_compare(
    dataset: str,
    path: Path | None,
    limit: int | None,
    classifier: str,
) -> None:
    records = load_dataset(dataset, path=path, limit=limit)
    train, test = split_records(records)
    label_names = sorted({r["label"] for r in records})

    pipeline = train_baseline(train, classifier)
    test_texts = [preprocess_text(r["text"]) for r in test]
    test_labels = [r["label"] for r in test]
    baseline_metrics = evaluate(pipeline, test_texts, test_labels)

    print("Running zero-shot transformer (may download model on first use)...")
    transformer_metrics = zero_shot_evaluate(test, label_names)
    comparison = compare_models(baseline_metrics, transformer_metrics)
    errors = error_analysis(test, baseline_metrics["predictions"])

    save_artifacts(
        pipeline,
        dataset,
        label_names,
        {"baseline": baseline_metrics, "comparison": comparison},
        classifier,
    )

    print(f"\n{'Model':<20} {'Accuracy':>10} {'F1 macro':>10}")
    print("-" * 42)
    print(f"{'Baseline (TF-IDF)':<20} {baseline_metrics['accuracy']:>10.3f} {baseline_metrics['f1_macro']:>10.3f}")
    print(f"{'Zero-shot HF':<20} {transformer_metrics['accuracy']:>10.3f} {transformer_metrics['f1_macro']:>10.3f}")
    print(f"\nWinner (F1 macro): {comparison['winner']}")
    print(f"\nBaseline misclassifications ({len(errors)}):")
    for err in errors[:5]:
        print(f"  [{err['id']}] actual={err['actual']} pred={err['predicted']}")
        print(f"    {err['text'][:100]}...")
    if len(errors) > 5:
        print(f"  ... and {len(errors) - 5} more")


def run_serve(dataset: str, port: int) -> None:
    import uvicorn

    from api import create_app

    artifacts_path = ARTIFACTS_DIR / dataset
    if not (artifacts_path / "baseline.joblib").exists():
        raise FileNotFoundError(
            f"No artifacts for {dataset}. Run: python start.py --dataset {dataset} train"
        )
    app = create_app(artifacts_path)
    print(f"Serving {dataset} on http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inkwell Capstone - Open Your Case")
    parser.add_argument(
        "command",
        choices=["explore", "train", "compare", "serve", "list"],
        nargs="?",
        default="list",
    )
    parser.add_argument("--dataset", default="movie_reviews")
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--classifier", choices=["nb", "lr", "svm"], default="lr")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.command == "list":
        print("Available datasets:")
        for key in list_datasets():
            print(f"  - {key}")
        return

    if args.command == "explore":
        run_explore(args.dataset, args.path, args.limit)
    elif args.command == "train":
        run_train(args.dataset, args.path, args.limit, args.classifier)
    elif args.command == "compare":
        run_compare(args.dataset, args.path, args.limit, args.classifier)
    elif args.command == "serve":
        run_serve(args.dataset, args.port)


if __name__ == "__main__":
    main()
