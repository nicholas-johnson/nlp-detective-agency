"""
Exercise 01 - Archive Themes (solution)
"""

import json
from pathlib import Path

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

RANDOM_STATE = 42


def load_archive(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def build_dtm(texts: list[str]):
    vectorizer = CountVectorizer(stop_words="english", min_df=2)
    dtm = vectorizer.fit_transform(texts)
    return dtm, vectorizer.get_feature_names_out()


def fit_lda(dtm, n_topics: int = 4) -> LatentDirichletAllocation:
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        max_iter=20,
    )
    lda.fit(dtm)
    return lda


def top_words(model, feature_names, n: int = 8) -> list[dict]:
    topics = []
    for topic_id, row in enumerate(model.components_):
        ranked = row.argsort()[::-1][:n]
        words = [(feature_names[i], round(float(row[i]), 4)) for i in ranked]
        topics.append({"topic_id": topic_id, "words": words})
    return topics


def dominant_topics(model, dtm) -> list[tuple[int, float]]:
    doc_topics = model.transform(dtm)
    return [(int(row.argmax()), round(float(row.max()), 4)) for row in doc_topics]


def archive_report(records: list[dict], n_topics: int = 4) -> dict:
    texts = [r["summary"] for r in records]
    dtm, feature_names = build_dtm(texts)
    lda = fit_lda(dtm, n_topics)
    topics = top_words(lda, feature_names)
    assignments = dominant_topics(lda, dtm)

    cases = [
        {
            "id": r["id"],
            "case_id": r["case_id"],
            "title": r["title"],
            "dominant_topic": topic_id,
            "weight": weight,
        }
        for r, (topic_id, weight) in zip(records, assignments, strict=True)
    ]

    return {"topics": topics, "cases": cases}


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "cold_cases.json"


def main() -> None:
    records = load_archive(DATA_PATH)
    report = archive_report(records)

    print("Inkwell Investigations - Archive Themes (LDA)")
    print("=" * 44)

    for topic in report["topics"]:
        words = ", ".join(term for term, _ in topic["words"][:6])
        print(f"\nTopic {topic['topic_id']}: {words}")

    print("\nCase assignments:")
    for case in report["cases"]:
        print(
            f"  {case['id']} ({case['case_id']}) - "
            f"topic {case['dominant_topic']} ({case['weight']:.3f})"
        )


if __name__ == "__main__":
    main()
