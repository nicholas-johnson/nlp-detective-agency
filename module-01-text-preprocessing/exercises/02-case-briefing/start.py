"""
Exercise 02 — Case Briefing
Detectives working a case need a word-frequency briefing from witness statements.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOPS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def normalize_text(text: str) -> str:
    """Lowercase, remove [REDACTED], strip CASE-XXX refs, collapse whitespace."""
    # TODO: apply the same cleaning rules as Exercise 01
    pass


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Return alphabetic tokens that are not English stopwords."""
    # TODO: filter tokens against STOPS
    pass


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatise alphabetic tokens (default noun POS)."""
    # TODO: lemmatise each alphabetic token
    pass


def preprocess_statement(text: str) -> list[str]:
    """Chain: normalize → word_tokenize → stopwords → lemmatize."""
    # TODO: return cleaned lemma tokens
    pass


def statements_for_case(statements: list[dict], case_id: str) -> list[dict]:
    """Return statements matching the given case_id."""
    # TODO: filter statements by case_id
    pass


def term_frequencies(texts: list[str]) -> Counter[str]:
    """Count preprocessed tokens across multiple texts (min length 3)."""
    # TODO: preprocess each text and count tokens with len >= 3
    pass


def case_briefing(
    statements: list[dict],
    case_id: str,
    top_n: int = 10,
) -> list[tuple[str, int]]:
    """Return the top_n most frequent terms for a case as (term, count) pairs."""
    # TODO: filter by case, compute frequencies, return most common
    pass


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"


def main() -> None:
    """Print a word-frequency briefing for one case."""
    case_id = sys.argv[1] if len(sys.argv) > 1 else input("Case ID (e.g. CASE-42): ").strip()
    if not case_id:
        case_id = "CASE-42"

    statements = json.loads(DATA_PATH.read_text())
    briefing = case_briefing(statements, case_id)

    print(f"\nInkwell Investigations — Briefing for {case_id}")
    print("=" * 40)
    if not briefing:
        print("No statements found for that case.")
        return

    for rank, (term, count) in enumerate(briefing, 1):
        print(f"  {rank:>2}. {term:<20} {count}")


if __name__ == "__main__":
    main()
