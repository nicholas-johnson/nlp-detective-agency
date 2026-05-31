"""
Exercise 03 - Semantic Search (solution)
"""

import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"


def load_statements(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def embed_texts(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def cosine_pairs(embeddings: list[list[float]], ids: list[str], n: int = 5) -> list[dict]:
    matrix = np.array(embeddings)
    sims = cosine_similarity(matrix)
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pairs.append({
                "id_a": ids[i],
                "id_b": ids[j],
                "score": round(float(sims[i][j]), 4),
            })
    return sorted(pairs, key=lambda p: p["score"], reverse=True)[:n]


def _tfidf_pairs(records: list[dict], n: int = 5) -> list[dict]:
    ids = [r["id"] for r in records]
    texts = [r["raw_text"] for r in records]
    matrix = TfidfVectorizer(stop_words="english").fit_transform(texts)
    sims = cosine_similarity(matrix)
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pairs.append({
                "id_a": ids[i],
                "id_b": ids[j],
                "score": round(float(sims[i][j]), 4),
            })
    return sorted(pairs, key=lambda p: p["score"], reverse=True)[:n]


def compare_with_tfidf(records: list[dict], n: int = 5) -> dict:
    texts = [r["raw_text"] for r in records]
    ids = [r["id"] for r in records]
    embeddings = embed_texts(texts)
    return {
        "embedding_pairs": cosine_pairs(embeddings, ids, n=n),
        "tfidf_pairs": _tfidf_pairs(records, n=n),
    }


def search_report(records: list[dict]) -> dict:
    comparison = compare_with_tfidf(records)
    embed_set = {(p["id_a"], p["id_b"]) for p in comparison["embedding_pairs"]}
    tfidf_set = {(p["id_a"], p["id_b"]) for p in comparison["tfidf_pairs"]}
    only_embedding = embed_set - tfidf_set

    return {
        "embedding_pairs": comparison["embedding_pairs"],
        "tfidf_pairs": comparison["tfidf_pairs"],
        "only_in_embeddings": sorted(only_embedding),
    }


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this exercise.")
        return

    records = load_statements(DATA_PATH)
    report = search_report(records)

    print("Inkwell Investigations - Semantic Search")
    print("=" * 44)

    print("\nTop pairs (OpenAI embeddings):")
    for p in report["embedding_pairs"]:
        print(f"  {p['id_a']} <-> {p['id_b']}  ({p['score']:.3f})")

    print("\nTop pairs (TF-IDF):")
    for p in report["tfidf_pairs"]:
        print(f"  {p['id_a']} <-> {p['id_b']}  ({p['score']:.3f})")

    if report["only_in_embeddings"]:
        print("\nPairs found by embeddings but not TF-IDF:")
        for id_a, id_b in report["only_in_embeddings"]:
            print(f"  {id_a} <-> {id_b}")
    else:
        print("\nNo unique embedding pairs vs TF-IDF top-5.")


if __name__ == "__main__":
    main()
