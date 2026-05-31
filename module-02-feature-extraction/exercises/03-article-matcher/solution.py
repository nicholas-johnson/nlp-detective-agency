"""
Exercise 03 - Article Matcher (solution)
"""

import json
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_articles(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def build_tfidf_matrix(texts: list[str]):
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer.get_feature_names_out()


def similarity_matrix(tfidf_matrix):
    return cosine_similarity(tfidf_matrix)


def top_similar_pairs(
    sim_matrix,
    ids: list[str],
    n: int = 5,
) -> list[dict]:
    pairs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            pairs.append({
                "id_a": ids[i],
                "id_b": ids[j],
                "score": round(float(sim_matrix[i][j]), 4),
            })
    return sorted(pairs, key=lambda p: p["score"], reverse=True)[:n]


def same_category_rate(pairs: list[dict], records: list[dict]) -> float:
    cat_by_id = {r["id"]: r["category"] for r in records}
    if not pairs:
        return 0.0
    matches = sum(
        1 for p in pairs
        if cat_by_id[p["id_a"]] == cat_by_id[p["id_b"]]
    )
    return round(matches / len(pairs), 4)


def compare_ngram_vocab_sizes(texts: list[str]) -> dict[str, int]:
    unigram_vec = CountVectorizer(stop_words="english", ngram_range=(1, 1))
    bigram_vec = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    unigram_vec.fit(texts)
    bigram_vec.fit(texts)
    return {
        "unigram": len(unigram_vec.get_feature_names_out()),
        "bigram": len(bigram_vec.get_feature_names_out()),
    }


def article_audit(records: list[dict], n_pairs: int = 5) -> dict:
    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]
    matrix, _ = build_tfidf_matrix(texts)
    sim = similarity_matrix(matrix)
    pairs = top_similar_pairs(sim, ids, n=n_pairs)

    return {
        "total_articles": len(records),
        "vocab_sizes": compare_ngram_vocab_sizes(texts),
        "top_pairs": pairs,
        "same_category_rate": same_category_rate(pairs, records),
    }


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "newsgroups_articles.json"


def main() -> None:
    records = load_articles(DATA_PATH)
    report = article_audit(records)

    print("20 Newsgroups - Similarity Audit")
    print("=" * 40)
    print(f"Articles: {report['total_articles']}")
    print(f"Same-category rate (top pairs): {report['same_category_rate']:.0%}")

    sizes = report["vocab_sizes"]
    print(f"\nVocabulary: unigram={sizes['unigram']}, bigram={sizes['bigram']}")

    print("\nTop similar pairs:")
    for pair in report["top_pairs"]:
        print(f"  {pair['id_a']} <-> {pair['id_b']}  ({pair['score']:.3f})")


if __name__ == "__main__":
    main()
