"""
Exercise 01 - Statement Audit
The chief wants a quality report on every witness statement before analysis begins.
"""

import json
import re
from pathlib import Path

from nltk.tokenize import sent_tokenize, word_tokenize


def load_statements(path: Path) -> list[dict]:
    """Read witness statements from a JSON file."""
    # TODO: load JSON from path and return the list of statement dicts
    pass


def normalize_text(text: str) -> str:
    """Lowercase, remove [REDACTED], strip CASE-XXX refs, collapse whitespace."""
    # TODO: apply regex cleaning and return normalised text
    pass


def tokenize_sentences(text: str) -> list[str]:
    """Return a list of sentences from normalised text."""
    # TODO: normalise text, then use NLTK sent_tokenize
    pass


def tokenize_words(text: str) -> list[str]:
    """Return a list of word tokens from normalised text."""
    # TODO: normalise text, then use NLTK word_tokenize
    pass


def audit_statement(statement: dict) -> dict:
    """
    Return an audit record for one statement:
    {"id", "witness", "sentence_count", "word_count", "needs_review"}

    needs_review is True when sentence_count > 4 OR word_count > 120.
    Count only alphabetic tokens for word_count.
    """
    # TODO: compute counts from statement["raw_text"] and set needs_review
    pass


def audit_archive(statements: list[dict]) -> list[dict]:
    """Audit every statement; return results sorted by id."""
    # TODO: map audit_statement across statements and sort by id
    pass


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"


def main() -> None:
    """Load the archive and print an audit report for the chief."""
    statements = load_statements(DATA_PATH)
    audits = audit_archive(statements)

    print("Inkwell Investigations - Statement Audit")
    print("=" * 72)
    print(f"{'ID':<10} {'Witness':<22} {'Sents':>5} {'Words':>5}  Review")
    print("-" * 72)
    for audit in audits:
        flag = "YES" if audit["needs_review"] else ""
        print(
            f"{audit['id']:<10} {audit['witness']:<22} "
            f"{audit['sentence_count']:>5} {audit['word_count']:>5}  {flag}"
        )

    flagged = sum(1 for a in audits if a["needs_review"])
    print("-" * 72)
    print(f"{len(audits)} statements audited, {flagged} flagged for review.")


if __name__ == "__main__":
    main()
