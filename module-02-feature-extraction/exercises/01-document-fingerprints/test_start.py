"""Tests for Exercise 01 — Document Fingerprints."""

from pathlib import Path

import pytest

from start import (
    build_bow_matrix,
    fingerprint_report,
    load_statements,
    texts_for_case,
    top_terms,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
STATEMENTS_PATH = DATA_DIR / "inkwell" / "statements.json"


@pytest.fixture()
def statements():
    return load_statements(STATEMENTS_PATH)


class TestTextsForCase:
    def test_case42_has_four_statements(self, statements):
        doc_ids, texts = texts_for_case(statements, "CASE-42")
        assert len(doc_ids) == 4
        assert len(texts) == 4
        assert all(isinstance(t, str) for t in texts)

    def test_unknown_case_empty(self, statements):
        doc_ids, texts = texts_for_case(statements, "CASE-99")
        assert doc_ids == []
        assert texts == []


class TestBowMatrix:
    def test_matrix_shape(self, statements):
        _, texts = texts_for_case(statements, "CASE-42")
        matrix, feature_names = build_bow_matrix(texts)
        assert matrix.shape[0] == 4
        assert matrix.shape[1] == len(feature_names)
        assert matrix.shape[1] > 0

    def test_top_terms_returns_counts(self, statements):
        _, texts = texts_for_case(statements, "CASE-42")
        matrix, feature_names = build_bow_matrix(texts)
        terms = top_terms(matrix, feature_names, 0, n=3)
        assert len(terms) <= 3
        assert all(isinstance(t, str) and isinstance(c, int) and c > 0 for t, c in terms)


class TestFingerprintReport:
    def test_report_length(self, statements):
        report = fingerprint_report(statements, "CASE-42")
        assert len(report) == 4

    def test_report_sorted_by_id(self, statements):
        report = fingerprint_report(statements, "CASE-42")
        ids = [r["id"] for r in report]
        assert ids == sorted(ids)

    def test_whitfield_has_dock(self, statements):
        report = fingerprint_report(statements, "CASE-42")
        whitfield = next(r for r in report if r["id"] == "STM-002")
        terms = [t for t, _ in whitfield["top_terms"]]
        assert "dock" in terms or "docks" in terms

    def test_unknown_case_empty_report(self, statements):
        assert fingerprint_report(statements, "CASE-99") == []

    def test_card_shape(self, statements):
        report = fingerprint_report(statements, "CASE-42")
        for card in report:
            assert "id" in card
            assert "witness" in card
            assert "top_terms" in card
            assert len(card["top_terms"]) > 0
