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
        print("Loading GloVe (first run downloads ~66MB, then loads from cache)...")
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
        print("6. PCA projection - GloVe word clusters")
        print("0. Quit")

        choice = input("\nChoice: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            model = get_w2v_model()
            print(f"\nVocabulary size: {len(model.wv)}")
            print(f"Vector dimensions: {model.vector_size}")
            print(f"\n{model.vector_size} dimensions means each word is a point in")
            print(f"{model.vector_size}-dimensional space. Words that appear in similar")
            print("contexts in the Inkwell case files will be nearby.")

        elif choice == "2":
            model = get_w2v_model()
            word = input("Word: ").strip().lower()
            if word not in model.wv:
                print(f"'{word}' not in vocabulary.")
                continue
            print("\nCosine similarity: 1.0 = identical direction, 0.0 = unrelated.")
            print("Scores above ~0.5 indicate meaningful similarity.\n")
            print(f"Nearest to '{word}':")
            for neighbour, score in model.wv.most_similar(word, topn=5):
                print(f"  {neighbour:<20} {score:.3f}")
            print("\nThese neighbours share context windows in the Inkwell case files.")

        elif choice == "3":
            model = get_w2v_model()
            print("\nVector arithmetic: positive words pull toward their direction,")
            print("negative words push away. Classic: king - man + woman = queen.\n")
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
            print("\nGloVe was trained on billions of Wikipedia tokens — it knows")
            print("general English but not Inkwell jargon. Compare with Word2Vec")
            print("(option 2) for domain-specific words.")

        elif choice == "5":
            print("\nAveraging word vectors gives one vector per document.")
            print("Works on short text; long mixed-topic documents average to mush.\n")
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
                        best_pair = (idxs[a], idxs[b], score)
            if best_pair:
                idx_a, idx_b, score = best_pair
                print(f"Most similar pair: {ids[idx_a]} <-> {ids[idx_b]}  (score: {score:.3f})")
                print(f"\n  {ids[idx_a]}: {texts[idx_a][:80]}...")
                print(f"  {ids[idx_b]}: {texts[idx_b][:80]}...")

        elif choice == "6":
            print("\nPCA projects 50 dimensions down to 2. Nearby points were")
            print("close in the original space, but distant points may have been")
            print("close in dimensions PCA discarded.\n")
            glove = get_glove_model()
            words = [
                "king", "queen", "prince", "princess",
                "dog", "cat", "horse", "fish",
                "paris", "france", "london", "england",
                "car", "bus", "train", "bicycle",
            ]
            words = [w for w in words if w in glove]
            matrix = np.array([glove[w] for w in words])
            coords = PCA(n_components=2).fit_transform(matrix)

            width, height = 60, 25
            xs = coords[:, 0]
            ys = coords[:, 1]
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            x_pad = (x_max - x_min) * 0.1 or 1.0
            y_pad = (y_max - y_min) * 0.1 or 1.0

            grid: dict[tuple[int, int], str] = {}
            for word, (x, y) in zip(words, coords, strict=True):
                col = int((x - x_min + x_pad) / (x_max - x_min + 2 * x_pad) * (width - 1))
                row = int((1 - (y - y_min + y_pad) / (y_max - y_min + 2 * y_pad)) * (height - 1))
                grid[(row, col)] = word

            for r in range(height):
                line = [" "] * width
                for c in range(width):
                    if (r, c) in grid:
                        label = grid[(r, c)]
                        for i, ch in enumerate(label):
                            if c + i < width:
                                line[c + i] = ch
                print("  |" + "".join(line) + "|")
            print("  +" + "-" * width + "+")
            print("\n  Clusters: royalty, animals, places, transport.")

        else:
            print("Unknown option.")


if __name__ == "__main__":
    main()
