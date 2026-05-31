"""
Inkwell Investigations - Evidence Prep Room (demo)
Run:  python module-01-text-preprocessing/demo/demo.py
"""

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "inkwell" / "statements.json"

STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()
STOPS = set(stopwords.words("english"))


def load_statements() -> list[dict]:
    return json.loads(DATA_PATH.read_text())


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = re.sub(r"\[redacted\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"case-\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t.isalpha() and t not in STOPS]


def stem_tokens(tokens: list[str]) -> list[str]:
    return [STEMMER.stem(t) for t in tokens if t.isalpha()]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    return [LEMMATIZER.lemmatize(t) for t in tokens if t.isalpha()]


def preprocess_statement(text: str) -> list[str]:
    text = normalize_text(text)
    tokens = word_tokenize(text)
    tokens = remove_stopwords(tokens)
    return lemmatize_tokens(tokens)


def pick_statement(statements: list[dict]) -> dict | None:
    print("\nStatements:")
    for i, s in enumerate(statements, 1):
        print(f"  {i}. {s['id']} - {s['witness']} ({s['case_id']})")
    choice = input("Pick a number: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(statements)):
        print("Invalid choice.")
        return None
    return statements[int(choice) - 1]


def archive_briefing(statements: list[dict], top_n: int = 10) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for stmt in statements:
        counter.update(preprocess_statement(stmt["raw_text"]))
    return counter.most_common(top_n)


def main() -> None:
    statements = load_statements()
    print("Inkwell Investigations - Evidence Prep Room")
    print("=" * 44)

    while True:
        print("\n1. List statements")
        print("2. Tokenise sentences")
        print("3. Tokenise words")
        print("4. Clean and normalise")
        print("5. Remove stopwords")
        print("6. Stem vs lemmatise")
        print("7. Full pipeline")
        print("8. Archive briefing (all cases)")
        print("0. Quit")
        choice = input("\nChoice: ").strip()

        if choice == "0":
            break
        if choice == "1":
            for s in statements:
                print(f"  {s['id']}  {s['witness']:<20} {s['case_id']}")
            continue

        if choice == "8":
            print("\nTop terms across the full archive:")
            for term, count in archive_briefing(statements):
                print(f"  {term:<20} {count}")
            continue

        stmt = pick_statement(statements)
        if stmt is None:
            continue

        raw = stmt["raw_text"]
        norm = normalize_text(raw)

        if choice == "2":
            sentences = sent_tokenize(norm)
            print(f"\n{len(sentences)} sentence(s):")
            for i, sent in enumerate(sentences, 1):
                print(f"  {i}. {sent}")

        elif choice == "3":
            tokens = word_tokenize(norm)
            print(f"\n{len(tokens)} token(s), first 20:")
            print("  " + ", ".join(tokens[:20]))

        elif choice == "4":
            print("\nRAW:")
            print(f"  {raw[:120]}...")
            print("\nCLEANED:")
            print(f"  {norm[:120]}...")

        elif choice == "5":
            tokens = word_tokenize(norm)
            alpha = [t for t in tokens if t.isalpha()]
            filtered = remove_stopwords(alpha)
            print(f"\nBefore ({len(alpha)}): {', '.join(alpha[:15])}...")
            print(f"After  ({len(filtered)}): {', '.join(filtered[:15])}...")

        elif choice == "6":
            tokens = remove_stopwords(word_tokenize(norm))
            stems = stem_tokens(tokens)
            lemmas = lemmatize_tokens(tokens)
            print(f"\n{'TOKEN':<16} {'STEM':<16} LEMMA")
            for tok, stem, lemma in zip(tokens[:12], stems[:12], lemmas[:12], strict=True):
                print(f"{tok:<16} {stem:<16} {lemma}")

        elif choice == "7":
            result = preprocess_statement(raw)
            print(f"\nPipeline output ({len(result)} tokens):")
            print("  " + ", ".join(result[:25]) + ("..." if len(result) > 25 else ""))

        else:
            print("Unknown option.")


if __name__ == "__main__":
    main()
