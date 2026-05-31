"""Tests for Exercise 03 - Custom Entities (Part A)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("spacy")

from start import audit_coverage, build_custom_nlp, case_id_patterns, extract_with_rules

STATEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
)


class TestCaseIdPatterns:
    def test_patterns_non_empty(self):
        patterns = case_id_patterns()
        assert len(patterns) >= 1


class TestBuildCustomNlp:
    def test_finds_case_42(self):
        nlp = build_custom_nlp(case_id_patterns())
        doc = nlp("CASE-42 is none of my business.")
        labels = [ent.label_ for ent in doc.ents]
        texts = [ent.text for ent in doc.ents]
        assert "CASE_ID" in labels
        assert "CASE-42" in texts

    def test_stm_001(self):
        statements = json.loads(STATEMENTS_PATH.read_text())
        stm1 = next(s for s in statements if s["id"] == "STM-001")
        nlp = build_custom_nlp(case_id_patterns())
        doc = nlp(stm1["raw_text"])
        case_ents = [e.text for e in doc.ents if e.label_ == "CASE_ID"]
        assert "CASE-42" in case_ents


class TestAuditCoverage:
    def test_finds_most_case_ids(self):
        statements = json.loads(STATEMENTS_PATH.read_text())
        result = audit_coverage(statements)
        assert result["found_by_ruler"] >= result["statements_with_case_id"] - 1

    def test_extract_with_rules(self):
        nlp = build_custom_nlp(case_id_patterns())
        results = extract_with_rules(nlp, ["CASE-17 again.", "Nothing here."])
        assert any("CASE-17" in str(r) for r in results)
