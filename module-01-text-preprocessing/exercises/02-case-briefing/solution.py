"""
Exercise 02 - Case Briefing (solution)
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
    text = text.lower()
    text = re.sub(r"\[redacted\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"case-\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.isalpha() and t not in STOPS]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    return [LEMMATIZER.lemmatize(t) for t in tokens if t.isalpha()]


def preprocess_statement(text: str) -> list[str]:
    text = normalize_text(text)
    tokens = word_tokenize(text)
    tokens = remove_stopwords(tokens)
    return lemmatize_tokens(tokens)


def statements_for_case(statements: list[dict], case_id: str) -> list[dict]:
    return [s for s in statements if s["case_id"] == case_id]


def term_frequencies(texts: list[str]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts:
        for token in preprocess_statement(text):
            if len(token) >= 3:
                counter[token] += 1
    return counter


def case_briefing(
    statements: list[dict],
    case_id: str,
    top_n: int = 10,
) -> list[tuple[str, int]]:
    case_stmts = statements_for_case(statements, case_id)
    if not case_stmts:
        return []
    texts = [s["raw_text"] for s in case_stmts]
    return term_frequencies(texts).most_common(top_n)


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"


def main() -> None:
    """Print a word-frequency briefing for one case."""
    case_id = sys.argv[1] if len(sys.argv) > 1 else input("Case ID (e.g. CASE-42): ").strip()
    if not case_id:
        case_id = "CASE-42"

    statements = json.loads(DATA_PATH.read_text())
    briefing = case_briefing(statements, case_id)

    print(f"\nInkwell Investigations - Briefing for {case_id}")
    print("=" * 40)
    if not briefing:
        print("No statements found for that case.")
        return

    for rank, (term, count) in enumerate(briefing, 1):
        print(f"  {rank:>2}. {term:<20} {count}")


if __name__ == "__main__":
    main()
