"""
Exercise 03 - Custom Entities
Add EntityRuler patterns for case IDs and support-ticket reference IDs.
"""

import json
import re
from pathlib import Path

import spacy

STATEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
)
TICKETS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "public"
    / "support_tickets_sample.json"
)


def case_id_patterns() -> list[dict]:
    """EntityRuler patterns for CASE-\\d+ identifiers."""
    # TODO
    raise NotImplementedError


def ticket_patterns() -> list[dict]:
    """EntityRuler patterns for TKT-, ORD-, and REF- identifiers."""
    # TODO: ORDER may span multiple tokens (ORD-2024, -, 991)
    raise NotImplementedError


def build_custom_nlp(patterns: list[dict] | None = None):
    """Load spaCy and add entity_ruler before ner pipe."""
    # TODO
    raise NotImplementedError


def extract_with_rules(nlp, texts: list[str]) -> list[list[tuple[str, str]]]:
    """Run nlp on each text; return (text, label) entity tuples."""
    # TODO
    raise NotImplementedError


def audit_coverage(statements: list[dict]) -> dict:
    """Count statements with CASE- IDs vs those found by ruler."""
    # TODO: return statements_with_case_id, found_by_ruler
    raise NotImplementedError


def audit_ticket_refs(records: list[dict]) -> dict:
    """Count gold ticket refs vs those found by ticket_patterns ruler."""
    # TODO: return total_refs, found_by_ruler, recall (normalize punctuation)
    raise NotImplementedError


def run_inkwell() -> None:
    # TODO: print CASE ID coverage audit
    raise NotImplementedError


def run_real_world() -> None:
    # TODO: print ticket ref recall on support tickets sample
    raise NotImplementedError


def main() -> None:
    run_inkwell()
    # Uncomment below (and comment out run_inkwell) to run the real-world extension:
    # run_real_world()


if __name__ == "__main__":
    main()
