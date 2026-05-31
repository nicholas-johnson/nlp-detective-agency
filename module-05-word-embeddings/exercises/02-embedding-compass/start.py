"""
Exercise 02 - Embedding Compass (solution)
"""

import gensim.downloader as api


def load_pretrained(model_name: str = "glove-wiki-gigaword-50"):
    return api.load(model_name)


def similarity(model, word_a: str, word_b: str) -> float:
    return float(model.similarity(word_a, word_b))


def analogy(model, pos: list[str], neg: list[str]) -> str:
    result = model.most_similar(positive=pos, negative=neg, topn=1)
    return result[0][0]


def odd_one_out(model, words: list[str]) -> str:
    return model.doesnt_match(words)


def compass_report(model) -> dict:
    similarity_pairs = [
        ("king", "queen"),
        ("man", "woman"),
        ("dock", "pier"),
    ]
    analogies = [
        {"pos": ["king", "woman"], "neg": ["man"], "label": "king - man + woman"},
        {"pos": ["paris", "england"], "neg": ["france"], "label": "paris - france + england"},
    ]
    odd_words = ["dock", "pier", "ledger", "warehouse"]

    return {
        "similarities": [
            {"word_a": a, "word_b": b, "score": round(similarity(model, a, b), 4)}
            for a, b in similarity_pairs
        ],
        "analogies": [
            {
                "label": item["label"],
                "answer": analogy(model, item["pos"], item["neg"]),
            }
            for item in analogies
        ],
        "odd_one_out": {
            "words": odd_words,
            "answer": odd_one_out(model, odd_words),
        },
    }


def main() -> None:
    print("Loading GloVe (first run downloads ~66MB)...")
    model = load_pretrained()
    report = compass_report(model)

    print("\nEmbedding Compass - GloVe Report")
    print("=" * 40)

    print("\nSimilarity:")
    for item in report["similarities"]:
        print(f"  {item['word_a']} / {item['word_b']}: {item['score']:.3f}")

    print("\nAnalogies:")
    for item in report["analogies"]:
        print(f"  {item['label']} = {item['answer']}")

    odd = report["odd_one_out"]
    print(f"\nOdd one out from {odd['words']}: {odd['answer']}")


if __name__ == "__main__":
    main()
