"""
Exercise 03 - Real-World Topics (solution)
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

RANDOM_STATE = 42


def load_articles(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def build_dtm(texts: list[str]):
    vectorizer = CountVectorizer(stop_words="english", min_df=2)
    dtm = vectorizer.fit_transform(texts)
    return dtm, vectorizer.get_feature_names_out()


def build_tfidf(texts: list[str]):
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer.get_feature_names_out()


def fit_lda(dtm, n_topics: int = 4) -> LatentDirichletAllocation:
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        max_iter=20,
    )
    lda.fit(dtm)
    return lda


def top_words(model, feature_names, n: int = 8) -> list[list[str]]:
    topics = []
    for row in model.components_:
        ranked = row.argsort()[::-1][:n]
        topics.append([feature_names[i] for i in ranked])
    return topics


def dominant_topic(model, dtm) -> list[int]:
    doc_topics = model.transform(dtm)
    return [int(row.argmax()) for row in doc_topics]


def topic_purity(assignments: list[int], true_labels: list[str]) -> list[dict]:
    groups: dict[int, list[str]] = defaultdict(list)
    for topic_id, label in zip(assignments, true_labels, strict=True):
        groups[topic_id].append(label)

    results = []
    for topic_id in sorted(groups):
        counts = Counter(groups[topic_id])
        majority_cat, majority_count = counts.most_common(1)[0]
        total = len(groups[topic_id])
        results.append({
            "topic_id": topic_id,
            "majority_category": majority_cat,
            "count": majority_count,
            "total": total,
            "purity": round(majority_count / total, 4),
        })
    return results


def contingency_matrix(assignments: list[int], true_labels: list[str], n_topics: int) -> dict:
    categories = sorted(set(true_labels))
    cat_idx = {c: i for i, c in enumerate(categories)}
    matrix = [[0] * len(categories) for _ in range(n_topics)]
    for topic_id, label in zip(assignments, true_labels, strict=True):
        matrix[topic_id][cat_idx[label]] += 1
    return {"categories": categories, "matrix": matrix}


def topic_audit(records: list[dict], n_topics: int = 4) -> dict:
    texts = [r["text"] for r in records]
    true_labels = [r["category"] for r in records]

    dtm, feature_names = build_dtm(texts)
    lda = fit_lda(dtm, n_topics)
    word_lists = top_words(lda, feature_names)
    assignments = dominant_topic(lda, dtm)

    purity = topic_purity(assignments, true_labels)
    avg_purity = sum(p["purity"] * p["total"] for p in purity) / len(records)
    cm = contingency_matrix(assignments, true_labels, n_topics)

    doc_assignments = [
        {"id": r["id"], "category": r["category"], "topic": t}
        for r, t in zip(records, assignments, strict=True)
    ]

    return {
        "n_topics": n_topics,
        "topics": word_lists,
        "purity": purity,
        "avg_purity": round(avg_purity, 4),
        "contingency": cm,
        "assignments": doc_assignments,
    }


DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "newsgroups_sample.json"


def main() -> None:
    records = load_articles(DATA_PATH)
    report = topic_audit(records)

    print("Topic Modelling - 20 Newsgroups Audit")
    print("=" * 44)

    for i, words in enumerate(report["topics"]):
        print(f"\nTopic {i}: {', '.join(words[:6])}")

    print(f"\nPurity per topic:")
    for p in report["purity"]:
        print(f"  Topic {p['topic_id']}: {p['majority_category']} - {p['purity']:.0%} ({p['count']}/{p['total']})")

    print(f"\nAverage purity: {report['avg_purity']:.0%}")

    cm = report["contingency"]
    cats = cm["categories"]
    header = "          " + "  ".join(f"{c[:8]:>8}" for c in cats)
    print(f"\nContingency matrix (topic x category):\n{header}")
    for i, row in enumerate(cm["matrix"]):
        vals = "  ".join(f"{v:>8}" for v in row)
        print(f"  Topic {i}  {vals}")


if __name__ == "__main__":
    main()
