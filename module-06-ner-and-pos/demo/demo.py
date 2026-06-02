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
    print("POS = part-of-speech (NOUN, VERB, ...).  Dep = dependency relation")
    print("(nsubj = subject, dobj = object).  Head = the word this token depends on.\n")
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
    print("\nSVO = Subject-Verb-Object. We walk the dependency tree: find")
    print("nsubj tokens, follow to their head verb, then find the verb's dobj.\n")
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
    print("\nPronouns ('he', 'it') and passive voice make triples imperfect —")
    print("always inspect manually on critical sentences.")


def demo_evidence_board() -> None:
    nlp = get_nlp()
    print("\nNER scans every statement and collects persons, places, dates,")
    print("and orgs into one board.\n")
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
    total = 0
    for label in sorted(board.keys()):
        entities = sorted(board[label])
        total += len(entities)
        print(f"{label}: {', '.join(entities)}")
    print(f"\n{total} unique entities found across all statements.")
    print("Missing a name? The pre-trained model doesn't know Inkwell")
    print("jargon — see option 4 (EntityRuler) for custom patterns.")


def demo_entity_ruler() -> None:
    print("\nThe EntityRuler adds regex patterns (CASE-\\d+) before the")
    print("statistical NER pipe — catching IDs the model would miss.\n")
    nlp = get_ruler_nlp()
    statements = load_statements()
    print("--- EntityRuler: CASE IDs ---")
    found = 0
    for record in statements:
        doc = nlp(record["raw_text"])
        case_ents = [e.text for e in doc.ents if e.label_ == "CASE_ID"]
        if case_ents:
            found += len(case_ents)
            print(f"  {record['id']}: {case_ents}")
    print(f"\n{found} CASE_ID entities found across {len(statements)} statements.")


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
    print("\nOpen the HTML files in a browser to see dependency arcs")
    print("and entity highlights rendered visually.")


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
    if full_ms > 0:
        speedup = full_ms / lite_ms if lite_ms > 0 else float("inf")
        print(f"\nDisabling parser + NER gives ~{speedup:.1f}x speedup.")
    print("Useful when you only need POS tags on a large corpus.")


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
