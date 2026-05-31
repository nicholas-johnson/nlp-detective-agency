"""
Exercise 03 - Custom Entities (solution)
"""

import argparse
import json
import re
from pathlib import Path

import spacy

STATEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
)
TICKETS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "public"
    / "support_tickets_sample.json"
)


def case_id_patterns() -> list[dict]:
    return [{"label": "CASE_ID", "pattern": [{"TEXT": {"REGEX": r"CASE-\d+"}}]}]


def ticket_patterns() -> list[dict]:
    return [
        {"label": "TICKET", "pattern": [{"TEXT": {"REGEX": r"TKT-\d+"}}]},
        {
            "label": "ORDER",
            "pattern": [
                {"TEXT": {"REGEX": r"ORD-\d{4}"}},
                {"TEXT": "-"},
                {"TEXT": {"REGEX": r"\d+"}},
            ],
        },
        {"label": "REF", "pattern": [{"TEXT": {"REGEX": r"REF-[A-Z0-9]+"}}]},
    ]


def _norm_ref(text: str) -> str:
    return text.rstrip(".,;:").upper()


def build_custom_nlp(patterns: list[dict] | None = None):
    nlp = spacy.load("en_core_web_sm")
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.get_pipe("entity_ruler")
    ruler.add_patterns(patterns or case_id_patterns())
    return nlp


def extract_with_rules(nlp, texts: list[str]) -> list[list[tuple[str, str]]]:
    results = []
    for text in texts:
        doc = nlp(text)
        results.append([(ent.text, ent.label_) for ent in doc.ents])
    return results


def audit_coverage(statements: list[dict]) -> dict:
    nlp = build_custom_nlp(case_id_patterns())
    case_pattern = re.compile(r"CASE-\d+", re.IGNORECASE)
    statements_with_case = [s for s in statements if case_pattern.search(s["raw_text"])]
    found = 0
    for record in statements_with_case:
        doc = nlp(record["raw_text"])
        if any(ent.label_ == "CASE_ID" for ent in doc.ents):
            found += 1
    return {
        "statements_with_case_id": len(statements_with_case),
        "found_by_ruler": found,
    }


def audit_ticket_refs(records: list[dict]) -> dict:
    nlp = build_custom_nlp(ticket_patterns())
    total_refs = sum(len(r["refs"]) for r in records)
    found = 0
    for record in records:
        doc = nlp(record["text"])
        found_texts = [_norm_ref(ent.text) for ent in doc.ents]
        for ref in record["refs"]:
            ref_n = _norm_ref(ref["text"])
            if any(ref_n in t or t.startswith(ref_n) for t in found_texts):
                found += 1
    return {
        "total_refs": total_refs,
        "found_by_ruler": found,
        "recall": round(found / total_refs, 3) if total_refs else 0.0,
    }


def run_inkwell() -> None:
    statements = json.loads(STATEMENTS_PATH.read_text())
    nlp = build_custom_nlp(case_id_patterns())
    print("Inkwell Investigations - Custom Entity Audit")
    print("=" * 48)
    coverage = audit_coverage(statements)
    print(
        f"CASE IDs found: {coverage['found_by_ruler']}/"
        f"{coverage['statements_with_case_id']} statements"
    )
    print("\nSample extractions:")
    for record in statements[:3]:
        doc = nlp(record["raw_text"])
        case_ents = [(e.text, e.label_) for e in doc.ents if e.label_ == "CASE_ID"]
        if case_ents:
            print(f"  {record['id']}: {case_ents}")


def run_real_world() -> None:
    records = json.loads(TICKETS_PATH.read_text())
    print("Custom Entity Audit - Support Tickets Sample")
    print("=" * 48)
    result = audit_ticket_refs(records)
    print(f"Reference IDs found: {result['found_by_ruler']}/{result['total_refs']}")
    print(f"Recall: {result['recall']}")


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
