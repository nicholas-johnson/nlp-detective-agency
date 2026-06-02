"""
Capstone Exercise 03 - NLP Audit Dashboard (solution)
Run every NLP technique from the course against a corpus and produce a report.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import spacy
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_INKWELL = REPO_ROOT / "data" / "inkwell"
DATA_PUBLIC = REPO_ROOT / "data" / "public"

RANDOM_STATE = 42

_nlp = None
_sentiment_pipeline = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_sentiment():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        _sentiment_pipeline = pipeline("sentiment-analysis", truncation=True, max_length=512)
    return _sentiment_pipeline


def load_corpus(
    dataset: str = "inkwell",
    path: Path | None = None,
) -> list[dict]:
    """Load records as {id, text, label} dicts."""
    if dataset == "inkwell":
        raw = json.loads((DATA_INKWELL / "statements.json").read_text())
        return [
            {"id": r["id"], "text": r["raw_text"], "label": r["case_id"]}
            for r in raw
        ]

    if dataset == "movie_reviews":
        raw = json.loads((DATA_PUBLIC / "movie_reviews_sample.json").read_text())
        return [
            {"id": r["id"], "text": r["text"], "label": r.get("sentiment", r.get("label", ""))}
            for r in raw
        ]

    if dataset == "sms_spam":
        raw = json.loads((DATA_PUBLIC / "sms_spam_sample.json").read_text())
        return [
            {"id": r.get("id", f"SMS-{i:04d}"), "text": r["text"], "label": r["label"]}
            for i, r in enumerate(raw, 1)
        ]

    if dataset == "custom" and path is not None:
        raw = json.loads(path.read_text())
        return [
            {
                "id": str(r.get("id", f"DOC-{i:04d}")),
                "text": str(r["text"]),
                "label": str(r.get("label", "")),
            }
            for i, r in enumerate(raw, 1)
        ]

    raise ValueError(f"Unknown dataset: {dataset}")


def corpus_stats(records: list[dict]) -> dict:
    """Compute corpus-level statistics."""
    texts = [r["text"] for r in records]
    lengths = [len(t) for t in texts]
    tokens = []
    for t in texts:
        tokens.extend(re.findall(r"\w+", t.lower()))
    label_counts = dict(Counter(r["label"] for r in records if r.get("label")))
    return {
        "count": len(records),
        "avg_length": round(statistics.mean(lengths), 1) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "vocab_size": len(set(tokens)),
        "label_balance": label_counts,
    }


def discover_topics(texts: list[str], n_topics: int = 3, n_terms: int = 5) -> list[dict]:
    """Run NMF on TF-IDF features."""
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    nmf = NMF(n_components=min(n_topics, len(texts)), random_state=RANDOM_STATE)
    nmf.fit(tfidf)

    topics = []
    for idx, component in enumerate(nmf.components_):
        top_indices = component.argsort()[-n_terms:][::-1]
        top_terms = [str(feature_names[i]) for i in top_indices]
        topics.append({"topic_id": idx, "top_terms": top_terms})
    return topics


def sentiment_scan(texts: list[str]) -> dict:
    """Run HF sentiment-analysis pipeline on texts."""
    model = _get_sentiment()
    results = model(texts)
    distribution: dict[str, int] = Counter()
    scores = []
    for r in results:
        distribution[r["label"]] += 1
        scores.append(round(float(r["score"]), 3))
    return {"distribution": dict(distribution), "scores": scores}


def entity_census(texts: list[str]) -> dict[str, dict]:
    """Run spaCy NER across all texts."""
    nlp = _get_nlp()
    census: dict[str, set[str]] = defaultdict(set)
    for text in texts:
        doc = nlp(text)
        for ent in doc.ents:
            label = ent.label_
            if label in ("GPE", "FAC"):
                label = "LOC"
            census[label].add(ent.text)
    return {
        label: {"entities": sorted(names), "count": len(names)}
        for label, names in sorted(census.items())
    }


def classification_probe(records: list[dict]) -> dict | None:
    """Train TF-IDF + LR baseline if labels exist."""
    labels = [r["label"] for r in records if r.get("label")]
    if not labels or len(set(labels)) < 2:
        return None

    texts = [r["text"].lower() for r in records]
    label_counts = Counter(labels)
    min_class_size = min(label_counts.values())

    if min_class_size < 2 or len(records) < 4:
        return None

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.25, random_state=RANDOM_STATE, stratify=labels,
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(train_texts, train_labels)
    preds = pipeline.predict(test_texts)
    return {
        "accuracy": round(float(accuracy_score(test_labels, preds)), 3),
        "f1_macro": round(float(f1_score(test_labels, preds, average="macro", zero_division=0)), 3),
        "labels": sorted(set(labels)),
        "train_size": len(train_texts),
        "test_size": len(test_texts),
    }


def build_report(
    dataset: str,
    stats: dict,
    topics: list[dict],
    sentiment: dict | None,
    entities: dict,
    classification: dict | None,
) -> dict:
    """Assemble a complete JSON-serialisable audit report."""
    return {
        "dataset": dataset,
        "corpus_stats": stats,
        "topics": topics,
        "sentiment": sentiment,
        "entity_census": entities,
        "classification": classification,
    }


def print_report(report: dict) -> None:
    """Print a formatted CLI summary."""
    print(f"\nNLP Audit Report - {report['dataset']}")
    print("=" * 40)

    stats = report["corpus_stats"]
    print(f"\n1. Corpus stats")
    print(f"   Documents: {stats['count']}  |  "
          f"Avg length: {stats['avg_length']} chars  |  "
          f"Vocabulary: {stats['vocab_size']} unique tokens")
    if stats.get("label_balance"):
        balance = ", ".join(f"{k}: {v}" for k, v in stats["label_balance"].items())
        print(f"   Labels: {balance}")

    topics = report["topics"]
    if topics:
        print(f"\n2. Topic discovery ({len(topics)} topics)")
        for t in topics:
            terms = ", ".join(t["top_terms"])
            print(f"   Topic {t['topic_id']}: {terms}")

    sentiment = report.get("sentiment")
    if sentiment:
        print(f"\n3. Sentiment scan")
        dist = sentiment["distribution"]
        total = sum(dist.values())
        parts = []
        for label, count in sorted(dist.items()):
            pct = round(100 * count / total) if total else 0
            parts.append(f"{label}: {count} ({pct}%)")
        print(f"   {' | '.join(parts)}")
    else:
        print(f"\n3. Sentiment scan (skipped - torch not available)")

    entities = report["entity_census"]
    if entities:
        print(f"\n4. Entity census")
        for label, data in entities.items():
            names = ", ".join(data["entities"][:5])
            suffix = f", ..." if data["count"] > 5 else ""
            print(f"   {label} ({data['count']}): {names}{suffix}")

    classification = report.get("classification")
    if classification:
        print(f"\n5. Classification probe")
        print(f"   Labels: {', '.join(classification['labels'])}")
        print(f"   Baseline F1: {classification['f1_macro']}  |  "
              f"Accuracy: {classification['accuracy']}")
        print(f"   Train: {classification['train_size']}  |  Test: {classification['test_size']}")
    else:
        print(f"\n5. Classification probe (skipped - insufficient labels)")

    print()


def save_report(report: dict, path: Path) -> None:
    """Write the report to a JSON file."""
    path.write_text(json.dumps(report, indent=2))
    print(f"Report saved to: {path}")


def run_audit(dataset: str, path: Path | None, n_topics: int) -> None:
    """Run full audit and print report."""
    records = load_corpus(dataset, path)
    texts = [r["text"] for r in records]

    stats = corpus_stats(records)
    topics = discover_topics(texts, n_topics=n_topics)

    sentiment = None
    try:
        sentiment = sentiment_scan(texts)
    except Exception:
        pass

    entities = entity_census(texts)
    classification = classification_probe(records)

    report = build_report(dataset, stats, topics, sentiment, entities, classification)
    print_report(report)
    save_report(report, Path(f"audit_{dataset}.json"))


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
