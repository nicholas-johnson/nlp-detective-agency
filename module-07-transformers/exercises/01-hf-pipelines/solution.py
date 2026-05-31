"""
Exercise 01 - HF Pipelines (solution)
"""

import argparse
import json
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


def load_pipeline(task: str, **kwargs):
    if task not in _pipelines:
        _pipelines[task] = pipeline(task, **kwargs)
    return _pipelines[task]


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def classify_sentiment(texts: list[str]) -> list[dict]:
    clf = load_pipeline("sentiment-analysis")
    results = clf(texts, truncation=True, max_length=512)
    if isinstance(texts, str):
        return [results]
    return results


def extract_entities_hf(texts: list[str]) -> list[list[dict]]:
    ner = load_pipeline("ner", grouped_entities=True)
    if isinstance(texts, str):
        return [ner(texts)]
    return [ner(t) for t in texts]


def zero_shot_classify(texts: list[str], labels: list[str]) -> list[dict]:
    clf = load_pipeline("zero-shot-classification")
    if isinstance(texts, str):
        texts = [texts]
    return clf(texts, candidate_labels=labels, truncation=True)


def _normalize(text: str) -> str:
    return text.lower().strip()


def compare_ner_to_gold(hf_entities: list[dict], gold_entities: list[dict]) -> dict:
    """Compare HF NER spans to gold entity list."""
    pred_texts = {_normalize(e.get("word", e.get("entity_group", ""))) for e in hf_entities}
    matched = sum(
        1
        for g in gold_entities
        if any(_normalize(g["text"]) in p or p in _normalize(g["text"]) for p in pred_texts)
    )
    return {
        "matched": matched,
        "gold_total": len(gold_entities),
        "recall": round(matched / len(gold_entities), 3) if gold_entities else 0.0,
    }


def run_inkwell() -> None:
    sentiment_records = load_json(SENTIMENT_PATH)
    statements = load_json(STATEMENTS_PATH)
    gold_by_id = {r["id"]: r["entities"] for r in load_json(GOLD_PATH)}

    print("Inkwell Investigations - HF Pipeline Lab")
    print("=" * 48)

    texts = [r["text"] for r in sentiment_records[:5]]
    sentiments = classify_sentiment(texts)
    print("\nSentiment (first 5 witness statements):")
    for rec, result in zip(sentiment_records[:5], sentiments, strict=True):
        label = result["label"]
        score = result["score"]
        print(f"  {rec['id']} ({rec['sentiment']}): {label} ({score:.3f})")

    stmt_texts = [s["raw_text"] for s in statements[:3]]
    entities = extract_entities_hf(stmt_texts)
    print("\nNER (first 3 statements):")
    for stmt, ents in zip(statements[:3], entities, strict=True):
        names = [e.get("entity_group", e.get("entity", "")) + ":" + e["word"] for e in ents]
        print(f"  {stmt['id']}: {names[:5]}")

    zero_labels = ["calm", "hostile"]
    zs = zero_shot_classify([sentiment_records[0]["text"]], zero_labels)[0]
    print(f"\nZero-shot on SNT-001: {zs['labels'][0]} ({zs['scores'][0]:.3f})")

    if statements:
        stm = statements[0]
        hf_ents = extract_entities_hf([stm["raw_text"]])[0]
        gold = gold_by_id.get(stm["id"], [])
        if gold:
            cmp = compare_ner_to_gold(hf_ents, gold)
            print(f"\nNER vs gold ({stm['id']}): recall={cmp['recall']}")


def run_real_world() -> None:
    sms = load_json(SMS_PATH)[:10]
    conll = load_json(CONLL_PATH)[:10]

    print("HF Pipeline Lab - Real-World Sample")
    print("=" * 48)

    sms_texts = [m["text"] for m in sms]
    sentiments = classify_sentiment(sms_texts)
    print("\nSMS sentiment (first 10):")
    for msg, result in zip(sms, sentiments, strict=True):
        print(f"  {msg['id']} ({msg['label']}): {result['label']} ({result['score']:.3f})")

    conll_texts = [c["text"] for c in conll]
    entities = extract_entities_hf(conll_texts)
    total_gold = sum(len(c["entities"]) for c in conll)
    total_matched = 0
    for record, hf_ents in zip(conll, entities, strict=True):
        cmp = compare_ner_to_gold(hf_ents, record["entities"])
        total_matched += cmp["matched"]
    recall = round(total_matched / total_gold, 3) if total_gold else 0.0
    print(f"\nCoNLL NER recall (first 10): {recall} ({total_matched}/{total_gold})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-world", action="store_true")
    args = parser.parse_args()
    if args.real_world:
        run_real_world()
    else:
        run_inkwell()


if __name__ == "__main__":
    main()
