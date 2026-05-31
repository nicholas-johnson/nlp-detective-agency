"""Tests for Exercise 03 - Custom Entities (Part B extension)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("spacy")

from start import audit_ticket_refs, build_custom_nlp, ticket_patterns

TICKETS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "public"
    / "support_tickets_sample.json"
)


class TestTicketExtension:
    def test_ticket_patterns_non_empty(self):
        patterns = ticket_patterns()
        assert len(patterns) >= 3

    def test_finds_tkt_8842(self):
        nlp = build_custom_nlp(ticket_patterns())
        doc = nlp("Please check ticket TKT-8842 for updates.")
        texts = [ent.text for ent in doc.ents]
        assert "TKT-8842" in texts

    def test_audit_ticket_refs(self):
        records = json.loads(TICKETS_PATH.read_text())
        result = audit_ticket_refs(records)
        assert result["total_refs"] > 50
        assert result["recall"] >= 0.9
