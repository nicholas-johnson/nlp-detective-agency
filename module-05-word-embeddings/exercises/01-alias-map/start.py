"""
Exercise 01 - Alias Map (solution)
"""

import json
import re
from pathlib import Path

from gensim.models import Word2Vec

STATEMENTS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
COLD_CASES_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "cold_cases.json"


def load_corpus(statements_path: Path, cold_cases_path: Path) -> list[str]:
    statements = json.loads(statements_path.read_text())
    cold_cases = json.loads(cold_cases_path.read_text())
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


def train_word2vec(
    sentences: list[list[str]],
    vector_size: int = 50,
    window: int = 5,
    min_count: int = 2,
) -> Word2Vec:
    return Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=1,
        seed=42,
    )


def nearest_neighbours(model: Word2Vec, word: str, n: int = 5) -> list[tuple[str, float]]:
    if word not in model.wv:
        return []
    return [(w, float(score)) for w, score in model.wv.most_similar(word, topn=n)]


def alias_map(model: Word2Vec, seed_words: list[str]) -> dict[str, list[tuple[str, float]]]:
    result = {}
    for word in seed_words:
        neighbours = nearest_neighbours(model, word, n=3)
        result[word] = neighbours
    return result


def main() -> None:
    texts = load_corpus(STATEMENTS_PATH, COLD_CASES_PATH)
    sentences = tokenize_corpus(texts)
    model = train_word2vec(sentences)

    seeds = ["dock", "warehouse", "accountant", "overcoat"]
    mapping = alias_map(model, seeds)

    print("Inkwell Investigations - Alias Map")
    print("=" * 40)
    print(f"Vocabulary: {len(model.wv)} words\n")

    for seed, neighbours in mapping.items():
        if not neighbours:
            print(f"{seed}: (not in vocabulary)")
            continue
        terms = ", ".join(f"{w} ({s:.2f})" for w, s in neighbours)
        print(f"{seed}: {terms}")


if __name__ == "__main__":
    main()
