"""
Exercise 02 - NER Extraction (solution)
"""

import json
from collections import defaultdict
from pathlib import Path

import spacy

STATEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
)
GOLD_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "inkwell"
    / "statement_entities.json"
)
CONLL_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "conll_ner_sample.json"
)

_nlp = None

LOC_LABELS = {"LOC", "GPE", "FAC"}
SPACY_TO_GOLD = {"GPE": "LOC", "FAC": "LOC", "LOC": "LOC", "PERSON": "PERSON", "DATE": "DATE", "TIME": "DATE"}


def load_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def load_statements(path: Path, case_id: str | None = None) -> list[dict]:
    records = json.loads(path.read_text())
    if case_id:
        records = [r for r in records if r["case_id"] == case_id]
    return records


def load_gold(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def extract_entities(doc) -> dict[str, list[str]]:
    by_label: dict[str, list[str]] = defaultdict(list)
    for ent in doc.ents:
        label = SPACY_TO_GOLD.get(ent.label_, ent.label_)
        if label in ("PERSON", "LOC", "DATE", "ORG", "MISC"):
            by_label[label].append(ent.text)
    return dict(by_label)


def build_evidence_board(statements: list[dict], case_id: str) -> dict[str, set[str]]:
    nlp = load_nlp()
    board: dict[str, set[str]] = defaultdict(set)
    for record in statements:
        if record["case_id"] != case_id:
            continue
        doc = nlp(record["raw_text"])
        entities = extract_entities(doc)
        for label, texts in entities.items():
            for text in texts:
                board[label].add(text)
    return dict(board)


def compare_witnesses(statements: list[dict], case_id: str) -> dict[str, dict[str, set[str]]]:
    nlp = load_nlp()
    per_witness: dict[str, dict[str, set[str]]] = {}
    for record in statements:
        if record["case_id"] != case_id:
            continue
        doc = nlp(record["raw_text"])
        entities = extract_entities(doc)
        per_witness[record["witness"]] = {k: set(v) for k, v in entities.items()}
    return per_witness


def _normalize(text: str) -> str:
    return text.lower().strip()


def _entity_match(pred_text: str, gold_text: str) -> bool:
    p, g = _normalize(pred_text), _normalize(gold_text)
    return g in p or p in g


def evaluate_ner(predicted: list[dict], gold: list[dict]) -> dict[str, dict[str, float]]:
    """predicted/gold: list of {text, label} dicts."""
    labels = sorted({g["label"] for g in gold} | {p["label"] for p in predicted})
    metrics: dict[str, dict[str, float]] = {}
    for label in labels:
        gold_items = [g for g in gold if g["label"] == label]
        pred_items = [p for p in predicted if p["label"] == label]
        tp = sum(
            1
            for g in gold_items
            if any(_entity_match(p["text"], g["text"]) for p in pred_items)
        )
        fp = sum(
            1
            for p in pred_items
            if not any(_entity_match(p["text"], g["text"]) for g in gold_items)
        )
        fn = len(gold_items) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        metrics[label] = {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "support": len(gold_items),
        }
    return metrics


def _doc_entities_list(doc) -> list[dict]:
    result = []
    for ent in doc.ents:
        label = ent.label_
        if label in ("GPE", "FAC"):
            label = "LOC"
        result.append({"text": ent.text, "label": label})
    return result


def run_inkwell() -> None:
    nlp = load_nlp()
    statements = load_statements(STATEMENTS_PATH)
    gold_records = {r["id"]: r["entities"] for r in load_gold(GOLD_PATH)}
    board = build_evidence_board(statements, "CASE-42")
    print("Inkwell Investigations - NER Evidence Board (CASE-42)")
    print("=" * 52)
    for label in sorted(board.keys()):
        items = sorted(board[label])
        print(f"{label}: {', '.join(items)}")
    print("\nGold-label evaluation (sample statements):")
    all_pred: list[dict] = []
    all_gold: list[dict] = []
    for record in statements:
        doc = nlp(record["raw_text"])
        all_pred.extend(
            {"text": e["text"], "label": e["label"]} for e in _doc_entities_list(doc)
        )
        all_gold.extend(gold_records.get(record["id"], []))
    metrics = evaluate_ner(all_pred, all_gold)
    for label, m in sorted(metrics.items()):
        if m["support"]:
            print(f"  {label:8} P={m['precision']:.3f} R={m['recall']:.3f} (n={m['support']})")


def run_real_world() -> None:
    nlp = load_nlp()
    samples = json.loads(CONLL_PATH.read_text())
    print("NER Audit - CoNLL-2003 Sample")
    print("=" * 52)
    all_pred: list[dict] = []
    all_gold: list[dict] = []
    for record in samples:
        doc = nlp(record["text"])
        for ent in doc.ents:
            label = ent.label_
            if label in ("GPE", "FAC"):
                label = "LOC"
            all_pred.append({"text": ent.text, "label": label})
        all_gold.extend(record["entities"])
    metrics = evaluate_ner(all_pred, all_gold)
    for label, m in sorted(metrics.items()):
        if m["support"]:
            print(f"  {label:8} P={m['precision']:.3f} R={m['recall']:.3f} (n={m['support']})")


def main() -> None:
    run_inkwell()
    # Uncomment below (and comment out run_inkwell) to run the real-world extension:
    # run_real_world()


if __name__ == "__main__":
    main()
