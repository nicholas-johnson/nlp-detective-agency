"""
Exercise 02 - NER Extraction
Extract persons, locations, and dates; build an evidence board.
"""

import argparse
import json
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


def load_nlp():
    """Load en_core_web_sm (cache in module-level _nlp)."""
    # TODO
    raise NotImplementedError


def load_statements(path: Path, case_id: str | None = None) -> list[dict]:
    """Load statements JSON; optionally filter by case_id."""
    # TODO
    raise NotImplementedError


def load_gold(path: Path) -> list[dict]:
    """Load gold entity annotations JSON."""
    # TODO
    raise NotImplementedError


def extract_entities(doc) -> dict[str, list[str]]:
    """Group doc.ents by label (map GPE/FAC to LOC)."""
    # TODO
    raise NotImplementedError


def build_evidence_board(statements: list[dict], case_id: str) -> dict[str, set[str]]:
    """Merge entities across all statements for a case."""
    # TODO
    raise NotImplementedError


def compare_witnesses(statements: list[dict], case_id: str) -> dict[str, dict[str, set[str]]]:
    """Per-witness entity sets for a case."""
    # TODO
    raise NotImplementedError


def evaluate_ner(predicted: list[dict], gold: list[dict]) -> dict[str, dict[str, float]]:
    """Per-label precision/recall; items are {text, label} dicts."""
    # TODO
    raise NotImplementedError


def run_inkwell() -> None:
    # TODO: print evidence board for CASE-42 and gold eval metrics
    raise NotImplementedError


def run_real_world() -> None:
    # TODO: evaluate NER on CoNLL sample
    raise NotImplementedError


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
