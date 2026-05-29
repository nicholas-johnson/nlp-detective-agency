"""
Exercise 01 — Statement Audit (solution)
"""

import json
import re
from pathlib import Path

from nltk.tokenize import sent_tokenize, word_tokenize


def load_statements(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[redacted\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"case-\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_sentences(text: str) -> list[str]:
    return sent_tokenize(normalize_text(text))


def tokenize_words(text: str) -> list[str]:
    return word_tokenize(normalize_text(text))


def audit_statement(statement: dict) -> dict:
    raw = statement["raw_text"]
    sentence_count = len(tokenize_sentences(raw))
    word_count = sum(1 for t in tokenize_words(raw) if t.isalpha())
    needs_review = sentence_count > 4 or word_count > 120
    return {
        "id": statement["id"],
        "witness": statement["witness"],
        "sentence_count": sentence_count,
        "word_count": word_count,
        "needs_review": needs_review,
    }


def audit_archive(statements: list[dict]) -> list[dict]:
    audits = [audit_statement(s) for s in statements]
    return sorted(audits, key=lambda a: a["id"])


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"


def main() -> None:
    """Load the archive and print an audit report for the chief."""
    statements = load_statements(DATA_PATH)
    audits = audit_archive(statements)

    print("Inkwell Investigations — Statement Audit")
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
