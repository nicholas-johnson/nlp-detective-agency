"""Tests for Exercise 01 - Grammar Audit (Part A)."""

from pathlib import Path

import pytest

pytest.importorskip("spacy")

from start import (
    audit_case,
    extract_svo_triples,
    load_nlp,
    load_statements,
    pos_summary,
    score_svo,
)

STATEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
)


@pytest.fixture(scope="module")
def nlp():
    return load_nlp()


class TestLoadNlp:
    def test_loads_model(self, nlp):
        doc = nlp("Test sentence.")
        assert len(doc) > 0


class TestPosSummary:
    def test_counts_verbs_and_nouns(self, nlp):
        doc = nlp("The detective saw the suspect near the docks.")
        summary = pos_summary(doc)
        assert summary["verb_count"] >= 1
        assert summary["noun_count"] >= 1
        assert "saw" in summary["verbs"] or "see" in [t.lower() for t in summary["verbs"]]


class TestSvoTriples:
    def test_extracts_triple(self, nlp):
        doc = nlp("She saw him near the station.")
        triples = extract_svo_triples(doc)
        assert len(triples) >= 1
        assert any(t["verb"] == "see" for t in triples)

    def test_score_svo_match(self):
        predicted = [{"subj": "She", "verb": "see", "obj": "him"}]
        gold = [{"subj": "She", "verb": "see", "obj": "him"}]
        result = score_svo(predicted, gold)
        assert result["matched"] == 1
        assert result["recall"] == 1.0


class TestAuditCase:
    def test_case_42_has_results(self):
        statements = load_statements(STATEMENTS_PATH, case_id="CASE-42")
        results = audit_case(statements, "CASE-42")
        assert len(results) == len(statements)
        assert all("svo" in r for r in results)
