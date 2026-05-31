"""Tests for Exercise 02 - Case Briefing."""

import json
from pathlib import Path

import pytest

from start import (
    case_briefing,
    preprocess_statement,
    remove_stopwords,
    statements_for_case,
    term_frequencies,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
STATEMENTS_PATH = DATA_DIR / "inkwell" / "statements.json"


@pytest.fixture()
def statements():
    return json.loads(STATEMENTS_PATH.read_text())


class TestPreprocess:
    def test_removes_stopwords(self):
        tokens = ["the", "dock", "was", "empty"]
        assert remove_stopwords(tokens) == ["dock", "empty"]

    def test_preprocess_returns_lemmas(self):
        result = preprocess_statement("The docks were EMPTY that night.")
        assert "dock" in result
        assert "the" not in result


class TestCaseFilter:
    def test_statements_for_case(self, statements):
        case42 = statements_for_case(statements, "CASE-42")
        assert len(case42) == 4
        assert all(s["case_id"] == "CASE-42" for s in case42)


class TestCaseBriefing:
    def test_case42_has_expected_terms(self, statements):
        briefing = case_briefing(statements, "CASE-42")
        terms = [t for t, _ in briefing]
        assert "dock" in terms
        assert "warehouse" in terms

    def test_no_stopwords_in_briefing(self, statements):
        briefing = case_briefing(statements, "CASE-42")
        terms = {t for t, _ in briefing}
        assert "the" not in terms
        assert "was" not in terms

    def test_unknown_case_returns_empty(self, statements):
        assert case_briefing(statements, "CASE-99") == []

    def test_respects_top_n(self, statements):
        briefing = case_briefing(statements, "CASE-42", top_n=3)
        assert len(briefing) == 3

    def test_sorted_by_frequency(self, statements):
        briefing = case_briefing(statements, "CASE-42")
        counts = [c for _, c in briefing]
        assert counts == sorted(counts, reverse=True)

    def test_min_token_length(self, statements):
        texts = ["I am ok at it."]
        freqs = term_frequencies(texts)
        assert all(len(term) >= 3 for term in freqs)
