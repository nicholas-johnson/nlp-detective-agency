"""Tests for Exercise 01 - Statement Audit."""

from pathlib import Path

import pytest

from start import (
    audit_archive,
    audit_statement,
    load_statements,
    normalize_text,
    tokenize_sentences,
    tokenize_words,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
STATEMENTS_PATH = DATA_DIR / "inkwell" / "statements.json"


class TestNormalizeText:
    def test_lowercases(self):
        assert "docks" in normalize_text("DOCKS")

    def test_removes_redacted(self):
        result = normalize_text("Saw [REDACTED] leave.")
        assert "redacted" not in result

    def test_removes_case_refs(self):
        result = normalize_text("Nothing to do with CASE-42 here.")
        assert "case-42" not in result

    def test_collapses_whitespace(self):
        assert normalize_text("too   many    spaces") == "too many spaces"


class TestTokenize:
    def test_sentences(self):
        sents = tokenize_sentences("First sentence. Second sentence.")
        assert len(sents) == 2

    def test_words_alpha_count(self):
        tokens = tokenize_words("Hello, world!")
        alpha = [t for t in tokens if t.isalpha()]
        assert alpha == ["hello", "world"]


class TestAuditStatement:
    def test_short_statement(self):
        stmt = {
            "id": "STM-TEST",
            "witness": "Test Witness",
            "raw_text": "I was home. Nothing to report.",
        }
        audit = audit_statement(stmt)
        assert audit["id"] == "STM-TEST"
        assert audit["witness"] == "Test Witness"
        assert audit["sentence_count"] == 2
        assert audit["needs_review"] is False

    def test_long_statement_flagged(self):
        stmt = {
            "id": "STM-LONG",
            "witness": "Frank Holloway",
            "raw_text": (
                "One. Two. Three. Four. Five. "
                + "word " * 130
            ),
        }
        audit = audit_statement(stmt)
        assert audit["needs_review"] is True

    def test_boundary_four_sentences_not_flagged(self):
        stmt = {
            "id": "STM-BOUND",
            "witness": "Boundary",
            "raw_text": "One witness spoke. A second agreed. A third left early. A fourth said nothing.",
        }
        audit = audit_statement(stmt)
        assert audit["sentence_count"] == 4
        assert audit["needs_review"] is False


class TestLoadAndArchive:
    def test_load_statements(self):
        statements = load_statements(STATEMENTS_PATH)
        assert len(statements) >= 8
        assert all("raw_text" in s for s in statements)

    def test_audit_archive_sorted(self):
        statements = load_statements(STATEMENTS_PATH)
        audits = audit_archive(statements)
        ids = [a["id"] for a in audits]
        assert ids == sorted(ids)
        assert len(audits) == len(statements)

    def test_holloway_needs_review(self):
        statements = load_statements(STATEMENTS_PATH)
        audits = audit_archive(statements)
        holloway = next(a for a in audits if a["id"] == "STM-009")
        assert holloway["needs_review"] is True
