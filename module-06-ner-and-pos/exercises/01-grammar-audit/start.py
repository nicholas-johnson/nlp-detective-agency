"""
Exercise 01 - Grammar Audit
POS-tag witness statements and extract subject-verb-object triples.
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
    """Load en_core_web_sm (cache in module-level _nlp)."""
    # TODO: load spacy model once and return it
    raise NotImplementedError


def load_statements(path: Path, case_id: str | None = None) -> list[dict]:
    """Load statements JSON; optionally filter by case_id."""
    # TODO
    raise NotImplementedError


def load_ud_sample(path: Path) -> list[dict]:
    """Load UD EWT sample JSON."""
    # TODO
    raise NotImplementedError


def pos_summary(doc) -> dict:
    """Return verb/noun counts and lists from a spaCy Doc."""
    # TODO: keys verb_count, noun_count, verbs, nouns
    raise NotImplementedError


def extract_svo_triples(doc) -> list[dict]:
    """Extract subject-verb-object triples via dependency relations (nsubj + dobj/attr)."""
    # TODO: return list of {subj, verb, obj} dicts (verb = lemma)
    raise NotImplementedError


def audit_case(statements: list[dict], case_id: str) -> list[dict]:
    """Run pos_summary and extract_svo_triples for each statement in case."""
    # TODO
    raise NotImplementedError


def score_svo(predicted: list[dict], gold: list[dict]) -> dict:
    """Score predicted triples against gold; return matched, total, recall."""
    # TODO: flexible match on verb lemma and subj/obj text
    raise NotImplementedError


def run_inkwell() -> None:
    # TODO: print grammar audit for CASE-42 (see README)
    raise NotImplementedError


def run_real_world() -> None:
    # TODO: score SVO on UD sample, print overall recall
    raise NotImplementedError


def main() -> None:
    run_inkwell()
    # Uncomment below (and comment out run_inkwell) to run the real-world extension:
    # run_real_world()


if __name__ == "__main__":
    main()
