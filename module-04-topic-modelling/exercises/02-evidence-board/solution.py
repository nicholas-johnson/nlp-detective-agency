"""
Exercise 02 - Evidence Board (solution)
"""

import json
from pathlib import Path

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RANDOM_STATE = 42

LABEL_KEYWORDS = {
    "Waterfront": {"dock", "docks", "warehouse", "pier", "quay", "jetty", "fog", "reeves", "waterfront"},
    "Financial": {"ledger", "accountant", "office", "receipt", "deposit", "invoice", "station", "fraud"},
    "Surveillance": {"motorcar", "overcoat", "plates", "vehicle", "grey", "mill", "road", "surveillance"},
    "Neighbourhood": {"neighbour", "shop", "street", "tram", "tenant", "landlord", "orchard", "garden"},
}


def load_archive(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def build_dtm(texts: list[str]):
    vectorizer = CountVectorizer(stop_words="english", min_df=2)
    dtm = vectorizer.fit_transform(texts)
    return dtm, vectorizer.get_feature_names_out()


def build_tfidf(texts: list[str]):
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer.get_feature_names_out()


def fit_lda(dtm, n_topics: int) -> LatentDirichletAllocation:
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        max_iter=20,
    )
    lda.fit(dtm)
    return lda


def _extract_top_words(model, feature_names, n: int = 8) -> list[list[str]]:
    topics = []
    for row in model.components_:
        ranked = row.argsort()[::-1][:n]
        topics.append([feature_names[i] for i in ranked])
    return topics


def perplexity_scores(dtm, k_values: list[int]) -> list[dict]:
    results = []
    for k in k_values:
        lda = fit_lda(dtm, k)
        results.append({"k": k, "perplexity": round(float(lda.perplexity(dtm)), 4)})
    return sorted(results, key=lambda r: r["k"])


def fit_nmf(tfidf_matrix, n_topics: int) -> NMF:
    nmf = NMF(n_components=n_topics, random_state=RANDOM_STATE, max_iter=200)
    nmf.fit(tfidf_matrix)
    return nmf


def compare_models(lda_model, nmf_model, feature_names) -> dict:
    return {
        "lda": _extract_top_words(lda_model, feature_names),
        "nmf": _extract_top_words(nmf_model, feature_names),
    }


def label_topics(top_words: list[dict]) -> dict[int, str]:
    labels = {}
    for topic in top_words:
        terms = {w.lower() for w, _ in topic["words"]}
        best_label = "General"
        best_score = 0
        for label, keywords in LABEL_KEYWORDS.items():
            score = len(terms & keywords)
            if score > best_score:
                best_score = score
                best_label = label
        labels[topic["topic_id"]] = best_label
    return labels


def evidence_board(records: list[dict], k_values: list[int] | None = None) -> dict:
    if k_values is None:
        k_values = [3, 4, 5, 6]

    texts = [r["summary"] for r in records]
    dtm, count_features = build_dtm(texts)
    tfidf, tfidf_features = build_tfidf(texts)

    scores = perplexity_scores(dtm, k_values)
    best_k = min(scores, key=lambda s: s["perplexity"])["k"]

    lda = fit_lda(dtm, best_k)
    nmf = fit_nmf(tfidf, best_k)

    lda_word_lists = _extract_top_words(lda, count_features)
    nmf_word_lists = _extract_top_words(nmf, tfidf_features)

    topic_dicts = [
        {"topic_id": i, "words": [(w, 1.0) for w in words]}
        for i, words in enumerate(lda_word_lists)
    ]
    topic_labels = label_topics(topic_dicts)

    doc_topics = lda.transform(dtm)
    dominant = doc_topics.argmax(axis=1)

    board_map: dict[int, list[str]] = {i: [] for i in range(best_k)}
    for record, topic_id in zip(records, dominant, strict=True):
        board_map[int(topic_id)].append(record["id"])

    board = [
        {"topic_label": topic_labels[topic_id], "case_ids": sorted(ids)}
        for topic_id, ids in board_map.items()
    ]

    return {
        "best_k": best_k,
        "perplexity_scores": scores,
        "lda_topics": lda_word_lists,
        "nmf_topics": nmf_word_lists,
        "board": board,
    }


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "cold_cases.json"


def main() -> None:
    records = load_archive(DATA_PATH)
    report = evidence_board(records)

    print("Inkwell Investigations - Evidence Board")
    print("=" * 44)
    print(f"\nBest topic count (lowest perplexity): k={report['best_k']}")

    print("\nLDA vs NMF top words:")
    for topic_id, lda_words in enumerate(report["lda_topics"]):
        nmf_words = report["nmf_topics"][topic_id]
        print(f"  Topic {topic_id}")
        print(f"    LDA: {', '.join(lda_words[:6])}")
        print(f"    NMF: {', '.join(nmf_words[:6])}")

    print("\nEvidence board:")
    for group in report["board"]:
        ids = ", ".join(group["case_ids"])
        print(f"  {group['topic_label']}: {ids}")


if __name__ == "__main__":
    main()
