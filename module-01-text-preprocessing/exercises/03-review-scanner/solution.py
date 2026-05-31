"""
Exercise 03 - Review Scanner (solution)
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
    return json.loads(path.read_text())


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.isalpha() and t not in STOPS]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    return [LEMMATIZER.lemmatize(t) for t in tokens if t.isalpha()]


def preprocess_review(text: str) -> list[str]:
    text = normalize_text(text)
    tokens = word_tokenize(text)
    tokens = remove_stopwords(tokens)
    return [t for t in lemmatize_tokens(tokens) if len(t) >= 3]


def class_term_frequencies(records: list[dict]) -> dict[str, Counter[str]]:
    freqs: dict[str, Counter[str]] = {"pos": Counter(), "neg": Counter()}
    for record in records:
        for token in preprocess_review(record["text"]):
            freqs[record["sentiment"]][token] += 1
    return freqs


def distinctive_terms(
    pos_freq: Counter[str],
    neg_freq: Counter[str],
    n: int = 10,
) -> list[tuple[str, float, str]]:
    """Return (term, ratio, dominant_class) sorted by ratio descending."""
    all_terms = set(pos_freq) | set(neg_freq)
    scored = []
    for term in all_terms:
        pos = pos_freq.get(term, 0) + 1
        neg = neg_freq.get(term, 0) + 1
        if pos >= neg:
            ratio = pos / neg
            dominant = "pos"
        else:
            ratio = neg / pos
            dominant = "neg"
        scored.append((term, round(ratio, 3), dominant))
    return sorted(scored, key=lambda x: x[1], reverse=True)[:n]


def review_audit(records: list[dict]) -> dict:
    freqs = class_term_frequencies(records)
    pos_top = freqs["pos"].most_common(10)
    neg_top = freqs["neg"].most_common(10)
    distinctive = distinctive_terms(freqs["pos"], freqs["neg"])
    vocab = set(freqs["pos"]) | set(freqs["neg"])
    token_counts = [len(preprocess_review(r["text"])) for r in records]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0

    return {
        "pos_top": pos_top,
        "neg_top": neg_top,
        "distinctive": distinctive,
        "total_vocab": len(vocab),
        "avg_tokens": round(avg_tokens, 1),
    }


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
