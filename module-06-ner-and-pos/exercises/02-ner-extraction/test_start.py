"""Tests for Exercise 02 - NER Extraction (Part A)."""

from pathlib import Path

import pytest

pytest.importorskip("spacy")

from start import (
    build_evidence_board,
    evaluate_ner,
    extract_entities,
    load_gold,
    load_nlp,
    load_statements,
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


@pytest.fixture(scope="module")
def nlp():
    return load_nlp()


class TestExtractEntities:
    def test_finds_person(self, nlp):
        doc = nlp("Margaret Hayes saw Reeves near the docks.")
        entities = extract_entities(doc)
        texts = " ".join(sum(entities.values(), [])).lower()
        assert "reeves" in texts or "margaret" in texts or "hayes" in texts


class TestEvidenceBoard:
    def test_case_42_board(self):
        statements = load_statements(STATEMENTS_PATH)
        board = build_evidence_board(statements, "CASE-42")
        assert len(board) > 0
        all_text = " ".join(
            t.lower() for texts in board.values() for t in texts
        )
        assert "reeves" in all_text or "dock" in all_text


class TestEvaluateNer:
    def test_perfect_match(self):
        pred = [{"text": "Reeves", "label": "PERSON"}]
        gold = [{"text": "Reeves", "label": "PERSON"}]
        metrics = evaluate_ner(pred, gold)
        assert metrics["PERSON"]["precision"] == 1.0
        assert metrics["PERSON"]["recall"] == 1.0

    def test_gold_file_loads(self):
        gold = load_gold(GOLD_PATH)
        assert len(gold) == 10
