"""
Exercise 01 - Transformer Inference Lab (solution)
"""

import argparse
import json
import time
from pathlib import Path

from transformers import pipeline

SENTIMENT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "inkwell"
    / "witness_sentiment.json"
)
STATEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
)
GOLD_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "inkwell"
    / "statement_entities.json"
)
SMS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "sms_spam_sample.json"
)
CONLL_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "conll_ner_sample.json"
)

_pipelines: dict = {}


def load_model(task: str, **kwargs):
    if task not in _pipelines:
        print(f"  Loading {task}...")
        _pipelines[task] = pipeline(task, **kwargs)
    return _pipelines[task]


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def analyse_sentiment(records: list[dict]) -> list[dict]:
    clf = load_model("sentiment-analysis")
    texts = [r["text"] for r in records]
    raw = clf(texts, truncation=True, max_length=512)
    results = []
    for rec, out in zip(records, raw, strict=True):
        results.append({
            "id": rec["id"],
            "witness": rec["witness"],
            "predicted": out["label"],
            "score": round(out["score"], 3),
            "actual": rec["sentiment"],
        })
    return results


def extract_entities(statements: list[dict]) -> list[dict]:
    ner = load_model("ner", grouped_entities=True)
    results = []
    for stmt in statements:
        ents = ner(stmt["raw_text"])
        results.append({
            "id": stmt["id"],
            "case_id": stmt["case_id"],
            "witness": stmt["witness"],
            "entities": [
                {"entity_group": e["entity_group"], "word": e["word"].strip()}
                for e in ents
            ],
        })
    return results


def build_evidence_board(entity_results: list[dict], case_id: str) -> dict:
    board: dict[str, set] = {}
    for result in entity_results:
        if result["case_id"] != case_id:
            continue
        for ent in result["entities"]:
            group = ent["entity_group"]
            board.setdefault(group, set()).add(ent["word"])
    return board


def classify_zero_shot(records: list[dict], labels: list[str]) -> list[dict]:
    clf = load_model("zero-shot-classification")
    results = []
    for rec in records:
        out = clf(rec["text"], candidate_labels=labels, truncation=True)
        results.append({
            "id": rec["id"],
            "predicted": out["labels"][0],
            "scores": {l: round(s, 3) for l, s in zip(out["labels"], out["scores"], strict=True)},
            "actual": rec["sentiment"],
        })
    return results


def summarise_longest(statements: list[dict]) -> dict:
    longest = max(statements, key=lambda s: len(s["raw_text"].split()))
    summarizer = load_model("summarization")
    out = summarizer(longest["raw_text"], max_length=60, min_length=20, do_sample=False)
    summary = out[0]["summary_text"]
    return {
        "id": longest["id"],
        "original_words": len(longest["raw_text"].split()),
        "summary": summary,
        "summary_words": len(summary.split()),
    }


def run_all() -> None:
    sentiment_records = load_json(SENTIMENT_PATH)
    statements = load_json(STATEMENTS_PATH)

    print("Inkwell Investigations - Transformer Inference Lab")
    print("=" * 52)
    total_start = time.time()

    # Sentiment
    t0 = time.time()
    sent_results = analyse_sentiment(sentiment_records)
    sent_time = time.time() - t0
    print(f"\n--- Sentiment Analysis ({len(sent_results)} statements) ---")
    for r in sent_results:
        print(f"  {r['id']} {r['witness']}: {r['predicted']} ({r['score']}) ← {r['actual']}")
    print(f"  Time: {sent_time:.1f}s")

    # NER
    t0 = time.time()
    ner_results = extract_entities(statements)
    ner_time = time.time() - t0
    print(f"\n--- Named Entity Recognition ({len(statements)} statements) ---")
    for r in ner_results:
        groups: dict[str, list] = {}
        for e in r["entities"]:
            groups.setdefault(e["entity_group"], []).append(e["word"])
        parts = [f"{k}: {', '.join(v)}" for k, v in groups.items()]
        print(f"  {r['id']}: {' | '.join(parts) or '(none)'}")
    board = build_evidence_board(ner_results, "CASE-42")
    print(f"  Evidence board (CASE-42): {dict({k: sorted(v) for k, v in board.items()})}")
    print(f"  Time: {ner_time:.1f}s")

    # Zero-shot
    t0 = time.time()
    zs_results = classify_zero_shot(sentiment_records, ["calm", "hostile"])
    zs_time = time.time() - t0
    print(f"\n--- Zero-shot Classification ---")
    correct = 0
    for r in zs_results:
        match = "✓" if r["predicted"] == r["actual"] else "✗"
        scores = " > ".join(f"{l} ({s})" for l, s in r["scores"].items())
        print(f"  {r['id']}: {scores}  actual={r['actual']} {match}")
        if r["predicted"] == r["actual"]:
            correct += 1
    print(f"  Accuracy: {correct}/{len(zs_results)}")
    print(f"  Time: {zs_time:.1f}s")

    # Summarisation
    t0 = time.time()
    summary = summarise_longest(statements)
    sum_time = time.time() - t0
    print(f"\n--- Summarisation (longest statement) ---")
    print(f"  {summary['id']} ({summary['original_words']} words → {summary['summary_words']} words):")
    print(f"  \"{summary['summary']}\"")
    print(f"  Time: {sum_time:.1f}s")

    total = time.time() - total_start
    print(f"\nTotal inference time: {total:.1f}s")


def run_real_world() -> None:
    sms = load_json(SMS_PATH)[:10]
    conll = load_json(CONLL_PATH)[:10]

    print("Transformer Inference Lab - Real-World Sample")
    print("=" * 48)

    clf = load_model("sentiment-analysis")
    sms_texts = [m["text"] for m in sms]
    sentiments = clf(sms_texts, truncation=True, max_length=512)
    print("\nSMS sentiment (first 10):")
    for msg, result in zip(sms, sentiments, strict=True):
        print(f"  {msg['id']} ({msg['label']}): {result['label']} ({result['score']:.3f})")

    ner = load_model("ner", grouped_entities=True)
    conll_texts = [c["text"] for c in conll]
    print(f"\nCoNLL NER (first 10):")
    for record in conll[:10]:
        ents = ner(record["text"])
        names = [f"{e['entity_group']}:{e['word']}" for e in ents[:5]]
        print(f"  {record['id']}: {names}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-world", action="store_true")
    args = parser.parse_args()
    if args.real_world:
        run_real_world()
    else:
        run_all()


if __name__ == "__main__":
    main()
