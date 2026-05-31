"""
Inkwell Investigations - Archive Room (demo)
Run:  python module-04-topic-modelling/demo/demo.py
"""

import json
from pathlib import Path

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "inkwell" / "cold_cases.json"
RANDOM_STATE = 42


def load_archive() -> list[dict]:
    return json.loads(DATA_PATH.read_text())


def build_dtm(texts: list[str]):
    vectorizer = CountVectorizer(stop_words="english", min_df=2)
    dtm = vectorizer.fit_transform(texts)
    return dtm, vectorizer


def build_tfidf(texts: list[str]):
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def fit_lda(dtm, n_topics: int) -> LatentDirichletAllocation:
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        max_iter=20,
    )
    lda.fit(dtm)
    return lda


def fit_nmf(tfidf_matrix, n_topics: int) -> NMF:
    nmf = NMF(n_components=n_topics, random_state=RANDOM_STATE, max_iter=200)
    nmf.fit(tfidf_matrix)
    return nmf


def print_top_words(model, feature_names, n_words: int = 8) -> None:
    for topic_id, row in enumerate(model.components_):
        ranked = row.argsort()[::-1][:n_words]
        terms = [feature_names[i] for i in ranked]
        print(f"  Topic {topic_id}: {', '.join(terms)}")


def pick_record(records: list[dict]) -> dict | None:
    for i, r in enumerate(records, 1):
        print(f"  {i}. {r['id']} - {r['title']}")
    choice = input("Pick a case: ").strip()
    if not choice.isdigit() or not (1 <= int(choice) <= len(records)):
        print("Invalid choice.")
        return None
    return records[int(choice) - 1]


def export_pyldavis(lda, dtm, vectorizer) -> None:
    try:
        import pyLDAvis
        import pyLDAvis.lda_model
    except ImportError:
        print("pyLDAvis not installed. Run: pip install -e \".[nlp]\"")
        return

    out = Path(__file__).resolve().parent / "archive_topics.html"
    vis = pyLDAvis.lda_model.prepare(lda, dtm, vectorizer)
    pyLDAvis.save_html(vis, str(out))
    print(f"Saved: {out}")


def main() -> None:
    records = load_archive()
    texts = [r["summary"] for r in records]

    while True:
        print("\nInkwell Investigations - Archive Room")
        print("=" * 40)
        print("\n1. List archive stats")
        print("2. LDA - top words per topic")
        print("3. Document-topic profile for one case")
        print("4. NMF - top words per topic")
        print("5. Perplexity sweep (k=3..6)")
        print("6. Export pyLDAvis HTML")
        print("0. Quit")

        choice = input("\nChoice: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            word_counts = [len(t.split()) for t in texts]
            avg = sum(word_counts) / len(word_counts)
            print(f"\nDocuments: {len(records)}")
            print(f"Average words per summary: {avg:.1f}")

        elif choice == "2":
            k = input("Number of topics (default 4): ").strip() or "4"
            dtm, vec = build_dtm(texts)
            lda = fit_lda(dtm, int(k))
            print(f"\nLDA top words (k={k}):")
            print_top_words(lda, vec.get_feature_names_out())

        elif choice == "3":
            k = input("Number of topics (default 4): ").strip() or "4"
            dtm, vec = build_dtm(texts)
            lda = fit_lda(dtm, int(k))
            doc_topics = lda.transform(dtm)
            record = pick_record(records)
            if record is None:
                continue
            idx = records.index(record)
            weights = doc_topics[idx]
            dominant = int(weights.argmax())
            print(f"\n{record['id']} - {record['title']}")
            print(f"Dominant topic: {dominant} (weight {weights[dominant]:.3f})")
            print("All topic weights:", ", ".join(f"{w:.3f}" for w in weights))

        elif choice == "4":
            k = input("Number of topics (default 4): ").strip() or "4"
            tfidf, vec = build_tfidf(texts)
            nmf = fit_nmf(tfidf, int(k))
            print(f"\nNMF top words (k={k}):")
            print_top_words(nmf, vec.get_feature_names_out())

        elif choice == "5":
            dtm, _ = build_dtm(texts)
            print(f"\n{'k':>4} {'perplexity':>12}")
            print("-" * 20)
            for k in range(3, 7):
                lda = fit_lda(dtm, k)
                print(f"{k:>4} {lda.perplexity(dtm):>12.2f}")

        elif choice == "6":
            k = input("Number of topics (default 4): ").strip() or "4"
            dtm, vec = build_dtm(texts)
            lda = fit_lda(dtm, int(k))
            export_pyldavis(lda, dtm, vec)

        else:
            print("Unknown option.")


if __name__ == "__main__":
    main()
