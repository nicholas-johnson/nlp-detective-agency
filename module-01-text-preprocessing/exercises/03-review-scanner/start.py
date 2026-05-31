"""
Exercise 03 - Review Scanner
Apply the preprocessing pipeline to real movie reviews and compare
positive vs negative vocabulary.
"""

import json
import re
from collections import Counter
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

STOPS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def load_reviews(path: Path) -> list[dict]:
    """Read movie review records from JSON."""
    # TODO: load and return list of dicts
    pass


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    # TODO: implement normalization for review text
    pass


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove English stopwords and non-alpha tokens."""
    # TODO
    pass


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatize alphabetic tokens."""
    # TODO
    pass


def preprocess_review(text: str) -> list[str]:
    """Full pipeline: normalize, tokenize, stopwords, lemmatize, min length 3."""
    # TODO
    pass


def class_term_frequencies(records: list[dict]) -> dict[str, Counter[str]]:
    """Return term frequency Counters for pos and neg reviews."""
    # TODO
    pass


def distinctive_terms(
    pos_freq: Counter[str],
    neg_freq: Counter[str],
    n: int = 10,
) -> list[tuple[str, float, str]]:
    """Return (term, ratio, dominant_class) for most distinctive terms."""
    # TODO: use smoothed ratio (count + 1) per class
    pass


def review_audit(records: list[dict]) -> dict:
    """
    Return {
        "pos_top": [(term, count), ...],
        "neg_top": [(term, count), ...],
        "distinctive": [(term, ratio, class), ...],
        "total_vocab": int,
        "avg_tokens": float,
    }
    """
    # TODO: full workflow
    pass


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "movie_reviews_sample.json"


def main() -> None:
    records = load_reviews(DATA_PATH)
    report = review_audit(records)

    print("Movie Reviews - Preprocessing Audit")
    print("=" * 40)
    print(f"Vocabulary size: {report['total_vocab']}")
    print(f"Average tokens per review: {report['avg_tokens']}")

    print("\nTop positive terms:")
    for term, count in report["pos_top"]:
        print(f"  {term:<20} {count}")

    print("\nTop negative terms:")
    for term, count in report["neg_top"]:
        print(f"  {term:<20} {count}")

    print("\nMost distinctive terms:")
    for term, ratio, dominant in report["distinctive"]:
        print(f"  {term:<20} ratio={ratio:.2f}  ({dominant})")


if __name__ == "__main__":
    main()
