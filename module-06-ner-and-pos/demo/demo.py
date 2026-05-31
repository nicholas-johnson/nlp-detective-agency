"""
Inkwell Investigations - Linguistic Lab (demo)
Run:  python module-06-ner-and-pos/demo/demo.py
"""

import json
import time
from pathlib import Path

import spacy
from spacy import displacy

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "inkwell"
STATEMENTS_PATH = DATA_DIR / "statements.json"

_nlp = None
_ruler_nlp = None


def get_nlp():
    global _nlp
    if _nlp is None:
        print("Loading en_core_web_sm...")
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def get_ruler_nlp():
    global _ruler_nlp
    if _ruler_nlp is None:
        nlp = spacy.load("en_core_web_sm")
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns([
            {"label": "CASE_ID", "pattern": [{"TEXT": {"REGEX": r"CASE-\d+"}}]},
        ])
        _ruler_nlp = nlp
    return _ruler_nlp


def load_statements() -> list[dict]:
    return json.loads(STATEMENTS_PATH.read_text())


def extract_svo(doc) -> list[tuple[str, str, str]]:
    triples = []
    for token in doc:
        if token.dep_ not in ("nsubj", "nsubjpass"):
            continue
        verb = token.head
        if verb.pos_ not in ("VERB", "AUX"):
            continue
        obj = next((c for c in verb.children if c.dep_ in ("dobj", "attr", "oprd")), None)
        if obj is not None:
            triples.append((token.text, verb.lemma_, obj.text))
    return triples


def demo_pipeline_tour() -> None:
    nlp = get_nlp()
    statements = load_statements()
    record = statements[0]
    doc = nlp(record["raw_text"])
    print(f"\n--- Pipeline tour: {record['id']} ---")
    print(f"{'Token':<12} {'POS':<6} {'Dep':<10} {'Head'}")
    print("-" * 45)
    for token in doc[:20]:
        print(f"{token.text:<12} {token.pos_:<6} {token.dep_:<10} {token.head.text}")
    if len(doc) > 20:
        print(f"  ... ({len(doc)} tokens total)")
    print("\nEntities:")
    for ent in doc.ents:
        print(f"  {ent.text:<20} {ent.label_}")


def demo_svo() -> None:
    nlp = get_nlp()
    case_id = input("Case ID (default CASE-42): ").strip() or "CASE-42"
    statements = [s for s in load_statements() if s["case_id"] == case_id]
    print(f"\n--- SVO triples: {case_id} ---")
    for record in statements:
        doc = nlp(record["raw_text"])
        triples = extract_svo(doc)
        print(f"\n{record['id']} {record['witness']}:")
        for s, v, o in triples[:5]:
            print(f"  ({s}, {v}, {o})")
        if len(triples) > 5:
            print(f"  ... +{len(triples) - 5} more")


def demo_evidence_board() -> None:
    nlp = get_nlp()
    case_id = input("Case ID (default CASE-42): ").strip() or "CASE-42"
    from collections import defaultdict

    board: dict[str, set[str]] = defaultdict(set)
    for record in load_statements():
        if record["case_id"] != case_id:
            continue
        doc = nlp(record["raw_text"])
        for ent in doc.ents:
            board[ent.label_].add(ent.text)
    print(f"\n--- Evidence board: {case_id} ---")
    for label in sorted(board.keys()):
        print(f"{label}: {', '.join(sorted(board[label]))}")


def demo_entity_ruler() -> None:
    nlp = get_ruler_nlp()
    statements = load_statements()
    print("\n--- EntityRuler: CASE IDs ---")
    for record in statements:
        doc = nlp(record["raw_text"])
        case_ents = [e.text for e in doc.ents if e.label_ == "CASE_ID"]
        if case_ents:
            print(f"  {record['id']}: {case_ents}")


def demo_displacy() -> None:
    nlp = get_nlp()
    statements = load_statements()
    record = statements[0]
    doc = nlp(record["raw_text"])
    out_dir = Path(__file__).resolve().parent
    dep_path = out_dir / "demo_deps.html"
    ent_path = out_dir / "demo_ents.html"
    html_dep = displacy.render(doc, style="dep", page=True, minify=True)
    html_ent = displacy.render(doc, style="ent", page=True, minify=True)
    dep_path.write_text(html_dep)
    ent_path.write_text(html_ent)
    print(f"\nSaved dependency viz: {dep_path}")
    print(f"Saved entity viz:       {ent_path}")


def demo_pipe_timing() -> None:
    nlp = get_nlp()
    text = load_statements()[8]["raw_text"]
    start = time.perf_counter()
    nlp(text)
    full_ms = (time.perf_counter() - start) * 1000
    with nlp.select_pipes(disable=["parser", "ner"]):
        start = time.perf_counter()
        nlp(text)
        lite_ms = (time.perf_counter() - start) * 1000
    print(f"\n--- Pipe timing ---")
    print(f"Full pipeline:  {full_ms:.1f} ms")
    print(f"Tagger only:    {lite_ms:.1f} ms")


def main() -> None:
    while True:
        print("\nInkwell Investigations - Linguistic Lab")
        print("=" * 40)
        print("\n1. Pipeline tour (tokens, POS, deps, ents)")
        print("2. SVO triples by case")
        print("3. NER evidence board")
        print("4. EntityRuler - CASE IDs")
        print("5. displacy export (HTML)")
        print("6. Pipe timing comparison")
        print("0. Quit")

        choice = input("\nChoice: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            demo_pipeline_tour()
        elif choice == "2":
            demo_svo()
        elif choice == "3":
            demo_evidence_board()
        elif choice == "4":
            demo_entity_ruler()
        elif choice == "5":
            demo_displacy()
        elif choice == "6":
            demo_pipe_timing()
        else:
            print("Unknown option.")


if __name__ == "__main__":
    main()
