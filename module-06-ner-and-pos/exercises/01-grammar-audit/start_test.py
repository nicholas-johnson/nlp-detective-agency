"""
Exercise 01 - Grammar Audit (solution)
"""

import json
from pathlib import Path

import spacy

STATEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
)
UD_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "ud_ewt_sample.json"

_nlp = None


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


def load_ud_sample(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def pos_summary(doc) -> dict:
    verbs = [t.text for t in doc if t.pos_ == "VERB"]
    nouns = [t.text for t in doc if t.pos_ in ("NOUN", "PROPN")]
    return {
        "verb_count": len(verbs),
        "noun_count": len(nouns),
        "verbs": verbs,
        "nouns": nouns,
    }


def extract_svo_triples(doc) -> list[dict]:
    triples = []
    for token in doc:
        if token.dep_ not in ("nsubj", "nsubjpass"):
            continue
        verb = token.head
        if verb.pos_ not in ("VERB", "AUX"):
            continue
        obj = None
        for child in verb.children:
            if child.dep_ in ("dobj", "attr", "oprd"):
                obj = child
                break
        if obj is not None:
            triples.append({
                "subj": token.text,
                "verb": verb.lemma_,
                "obj": obj.text,
            })
    return triples


def audit_case(statements: list[dict], case_id: str) -> list[dict]:
    nlp = load_nlp()
    results = []
    for record in statements:
        doc = nlp(record["raw_text"])
        results.append({
            "id": record["id"],
            "witness": record["witness"],
            "pos": pos_summary(doc),
            "svo": extract_svo_triples(doc),
        })
    return results


def _normalize(text: str) -> str:
    return text.lower().strip()


def _triple_matches(predicted: dict, gold: dict) -> bool:
    if _normalize(predicted["verb"]) != _normalize(gold["verb"]):
        return False
    subj_ok = (
        _normalize(gold["subj"]) in _normalize(predicted["subj"])
        or _normalize(predicted["subj"]) in _normalize(gold["subj"])
    )
    obj_ok = (
        _normalize(gold["obj"]) in _normalize(predicted["obj"])
        or _normalize(predicted["obj"]) in _normalize(gold["obj"])
    )
    return subj_ok and obj_ok


def score_svo(predicted: list[dict], gold: list[dict]) -> dict:
    if not gold:
        return {"matched": 0, "total": 0, "recall": 0.0}
    matched = sum(
        1
        for g in gold
        if any(_triple_matches(p, g) for p in predicted)
    )
    return {
        "matched": matched,
        "total": len(gold),
        "recall": round(matched / len(gold), 3),
    }


def run_inkwell() -> None:
    nlp = load_nlp()
    statements = load_statements(STATEMENTS_PATH, case_id="CASE-42")
    print("Inkwell Investigations - Grammar Audit (CASE-42)")
    print("=" * 48)
    for record in statements:
        doc = nlp(record["raw_text"])
        summary = pos_summary(doc)
        triples = extract_svo_triples(doc)
        print(f"\n{record['id']} {record['witness']}:")
        print(f"  Verbs ({summary['verb_count']}): {', '.join(summary['verbs'][:8])}")
        for t in triples[:5]:
            print(f"  SVO: ({t['subj']}, {t['verb']}, {t['obj']})")


def run_real_world() -> None:
    nlp = load_nlp()
    samples = load_ud_sample(UD_PATH)
    print("Grammar Audit - UD English EWT Sample")
    print("=" * 48)
    total_matched = 0
    total_gold = 0
    for record in samples:
        doc = nlp(record["text"])
        predicted = extract_svo_triples(doc)
        gold = record["gold_svo"]
        scores = score_svo(predicted, gold)
        total_matched += scores["matched"]
        total_gold += scores["total"]
    overall = round(total_matched / total_gold, 3) if total_gold else 0.0
    print(f"\nSVO recall (approximate): {overall} ({total_matched}/{total_gold} gold triples)")


def main() -> None:
    run_inkwell()
    # Uncomment below (and comment out run_inkwell) to run the real-world extension:
    # run_real_world()


if __name__ == "__main__":
    main()
