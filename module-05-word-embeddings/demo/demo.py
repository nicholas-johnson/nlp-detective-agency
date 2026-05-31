"""
Inkwell Investigations - Embedding Lab (demo)
Run:  python module-05-word-embeddings/demo/demo.py
"""

import json
import re
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "inkwell"
STATEMENTS_PATH = DATA_DIR / "statements.json"
COLD_CASES_PATH = DATA_DIR / "cold_cases.json"

_w2v_model: Word2Vec | None = None
_glove_model = None


def load_corpus() -> list[str]:
    statements = json.loads(STATEMENTS_PATH.read_text())
    cold_cases = json.loads(COLD_CASES_PATH.read_text())
    texts = [s["raw_text"] for s in statements]
    texts += [c["summary"] for c in cold_cases]
    return texts


def tokenize_corpus(texts: list[str]) -> list[list[str]]:
    sentences = []
    for text in texts:
        tokens = re.findall(r"[a-z]+", text.lower())
        tokens = [t for t in tokens if len(t) >= 3]
        if tokens:
            sentences.append(tokens)
    return sentences


def get_w2v_model() -> Word2Vec:
    global _w2v_model
    if _w2v_model is None:
        sentences = tokenize_corpus(load_corpus())
        _w2v_model = Word2Vec(
            sentences,
            vector_size=50,
            window=5,
            min_count=2,
            workers=1,
            seed=42,
        )
    return _w2v_model


def get_glove_model():
    global _glove_model
    if _glove_model is None:
        import gensim.downloader as api
        print("Loading GloVe (first time may download ~66MB)...")
        _glove_model = api.load("glove-wiki-gigaword-50")
    return _glove_model


def document_vector(model, text: str) -> np.ndarray | None:
    tokens = re.findall(r"[a-z]+", text.lower())
    vectors = [model[t] for t in tokens if t in model]
    if not vectors:
        return None
    return np.mean(vectors, axis=0)


def main() -> None:
    while True:
        print("\nInkwell Investigations - Embedding Lab")
        print("=" * 40)
        print("\n1. Train Word2Vec - show vocab size")
        print("2. Nearest neighbours (Word2Vec)")
        print("3. Vector arithmetic (Word2Vec)")
        print("4. Pre-trained GloVe - nearest neighbours")
        print("5. Document similarity (Word2Vec)")
        print("6. PCA projection - top words")
        print("0. Quit")

        choice = input("\nChoice: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            model = get_w2v_model()
            print(f"\nVocabulary size: {len(model.wv)}")
            print(f"Vector dimensions: {model.vector_size}")

        elif choice == "2":
            model = get_w2v_model()
            word = input("Word: ").strip().lower()
            if word not in model.wv:
                print(f"'{word}' not in vocabulary.")
                continue
            print(f"\nNearest to '{word}':")
            for neighbour, score in model.wv.most_similar(word, topn=5):
                print(f"  {neighbour:<20} {score:.3f}")

        elif choice == "3":
            model = get_w2v_model()
            pos = input("Positive words (comma-separated): ").strip().lower().split(",")
            neg = input("Negative words (comma-separated, or blank): ").strip().lower()
            neg = [w.strip() for w in neg.split(",") if w.strip()] if neg else []
            pos = [w.strip() for w in pos if w.strip()]
            try:
                results = model.wv.most_similar(positive=pos, negative=neg, topn=3)
                print("\nResults:")
                for word, score in results:
                    print(f"  {word:<20} {score:.3f}")
            except KeyError as e:
                print(f"Word not in vocabulary: {e}")

        elif choice == "4":
            glove = get_glove_model()
            word = input("Word: ").strip().lower()
            if word not in glove:
                print(f"'{word}' not in GloVe vocabulary.")
                continue
            print(f"\nGloVe nearest to '{word}':")
            for neighbour, score in glove.most_similar(word, topn=5):
                print(f"  {neighbour:<20} {score:.3f}")

        elif choice == "5":
            model = get_w2v_model()
            statements = json.loads(STATEMENTS_PATH.read_text())
            ids = [s["id"] for s in statements]
            texts = [s["raw_text"] for s in statements]
            vectors = [document_vector(model.wv, t) for t in texts]
            valid = [(i, v) for i, v in enumerate(vectors) if v is not None]
            if len(valid) < 2:
                print("Not enough valid document vectors.")
                continue
            idxs, vecs = zip(*valid, strict=True)
            sims = cosine_similarity(list(vecs))
            best_score = -1.0
            best_pair = None
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    score = float(sims[a][b])
                    if score > best_score:
                        best_score = score
                        best_pair = (ids[idxs[a]], ids[idxs[b]], score)
            if best_pair:
                id_a, id_b, score = best_pair
                print(f"\nMost similar pair: {id_a} <-> {id_b}  (score: {score:.3f})")

        elif choice == "6":
            model = get_w2v_model()
            words = list(model.wv.index_to_key)[:50]
            matrix = np.array([model.wv[w] for w in words])
            coords = PCA(n_components=2).fit_transform(matrix)
            print(f"\n{'Word':<15} {'x':>8} {'y':>8}")
            print("-" * 35)
            for word, (x, y) in zip(words[:10], coords[:10], strict=True):
                print(f"{word:<15} {x:>8.3f} {y:>8.3f}")

        else:
            print("Unknown option.")


if __name__ == "__main__":
    main()
