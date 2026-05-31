"""
Inkwell Investigations - Fingerprint Lab (demo)
Run:  python module-02-feature-extraction/demo/demo.py
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "inkwell" / "statements.json"


def load_statements() -> list[dict]:
    return json.loads(DATA_PATH.read_text())


def texts_for_case(statements: list[dict], case_id: str) -> tuple[list[str], list[str]]:
    matched = [s for s in statements if s["case_id"] == case_id]
    return [s["id"] for s in matched], [s["raw_text"] for s in matched]


def build_count_matrix(texts: list[str]):
    vectorizer = CountVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer.get_feature_names_out()


def build_tfidf_matrix(texts: list[str]):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer.get_feature_names_out()


def top_count_terms(matrix, feature_names, doc_index: int, n: int = 5) -> list[tuple[str, int]]:
    row = matrix[doc_index].toarray().flatten()
    ranked = row.argsort()[::-1]
    return [
        (feature_names[i], int(row[i]))
        for i in ranked
        if row[i] > 0
    ][:n]


def top_tfidf_terms(matrix, feature_names, doc_index: int, n: int = 5) -> list[tuple[str, float]]:
    row = matrix[doc_index].toarray().flatten()
    ranked = row.argsort()[::-1]
    return [
        (feature_names[i], round(float(row[i]), 3))
        for i in ranked
        if row[i] > 0
    ][:n]


def cosine_sim(matrix, i: int, j: int) -> float:
    sims = cosine_similarity(matrix)
    return float(sims[i][j])


def ngram_vocab_sizes(texts: list[str]) -> dict[str, int]:
    sizes = {}
    for label, ngram_range in [("(1,1) unigrams", (1, 1)), ("(1,2) +bigrams", (1, 2)), ("(1,3) +trigrams", (1, 3))]:
        vec = CountVectorizer(stop_words="english", ngram_range=ngram_range)
        vec.fit(texts)
        sizes[label] = len(vec.get_feature_names_out())
    return sizes


def pick_case(statements: list[dict]) -> str | None:
    cases = sorted({s["case_id"] for s in statements})
    print("\nCases:")
    for i, case_id in enumerate(cases, 1):
        count = sum(1 for s in statements if s["case_id"] == case_id)
        print(f"  {i}. {case_id} ({count} statements)")
    choice = input("Pick a number: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(cases)):
        print("Invalid choice.")
        return None
    return cases[int(choice) - 1]


def pick_statement(statements: list[dict], case_id: str) -> tuple[int, dict] | None:
    case_stmts = [s for s in statements if s["case_id"] == case_id]
    print(f"\nStatements for {case_id}:")
    for i, s in enumerate(case_stmts, 1):
        print(f"  {i}. {s['id']} - {s['witness']}")
    choice = input("Pick a number: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(case_stmts)):
        print("Invalid choice.")
        return None
    idx = int(choice) - 1
    return idx, case_stmts[idx]


def main() -> None:
    statements = load_statements()
    print("Inkwell Investigations - Fingerprint Lab")
    print("=" * 44)

    while True:
        print("\n1. List cases")
        print("2. BoW matrix shape for a case")
        print("3. BoW top terms for one statement")
        print("4. TF-IDF distinctive terms for one statement")
        print("5. N-gram vocabulary sizes for a case")
        print("6. Cosine similarity between two statements")
        print("0. Quit")
        choice = input("\nChoice: ").strip()

        if choice == "0":
            break

        if choice == "1":
            cases = sorted({s["case_id"] for s in statements})
            for case_id in cases:
                count = sum(1 for s in statements if s["case_id"] == case_id)
                print(f"  {case_id}: {count} statement(s)")
            continue

        case_id = pick_case(statements)
        if case_id is None:
            continue

        doc_ids, texts = texts_for_case(statements, case_id)
        if not texts:
            print("No statements for that case.")
            continue

        if choice == "2":
            matrix, names = build_count_matrix(texts)
            print(f"\nBoW matrix shape: {matrix.shape}")
            print(f"Vocabulary ({len(names)} terms): {', '.join(names[:12])}...")

        elif choice == "3":
            picked = pick_statement(statements, case_id)
            if picked is None:
                continue
            idx, stmt = picked
            matrix, names = build_count_matrix(texts)
            terms = top_count_terms(matrix, names, idx)
            print(f"\nTop count terms for {stmt['id']} ({stmt['witness']}):")
            for term, count in terms:
                print(f"  {term:<20} {count}")

        elif choice == "4":
            picked = pick_statement(statements, case_id)
            if picked is None:
                continue
            idx, stmt = picked
            matrix, names = build_tfidf_matrix(texts)
            terms = top_tfidf_terms(matrix, names, idx)
            print(f"\nDistinctive TF-IDF terms for {stmt['id']} ({stmt['witness']}):")
            for term, weight in terms:
                print(f"  {term:<20} {weight}")

        elif choice == "5":
            sizes = ngram_vocab_sizes(texts)
            print(f"\nVocabulary sizes for {case_id}:")
            for label, size in sizes.items():
                print(f"  {label:<20} {size}")

        elif choice == "6":
            if len(texts) < 2:
                print("Need at least 2 statements to compare.")
                continue
            print("Pick first statement:")
            first = pick_statement(statements, case_id)
            if first is None:
                continue
            print("Pick second statement:")
            second = pick_statement(statements, case_id)
            if second is None:
                continue
            i, stmt_a = first
            j, stmt_b = second
            matrix, _ = build_tfidf_matrix(texts)
            score = cosine_sim(matrix, i, j)
            print(f"\nSimilarity: {stmt_a['id']} <-> {stmt_b['id']} = {score:.3f}")

        else:
            print("Unknown option.")


if __name__ == "__main__":
    main()
