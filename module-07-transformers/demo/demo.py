"""
Inkwell Investigations - Transformer Lab (demo)
Run:  python module-07-transformers/demo/demo.py
"""

import json
from pathlib import Path

import tiktoken

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "inkwell"
STATEMENTS_PATH = DATA_DIR / "statements.json"
SENTIMENT_PATH = DATA_DIR / "witness_sentiment.json"

_pipelines: dict = {}


def load_statements() -> list[dict]:
    return json.loads(STATEMENTS_PATH.read_text())


def load_sentiment() -> list[dict]:
    return json.loads(SENTIMENT_PATH.read_text())


def get_pipeline(task: str, **kwargs):
    if task not in _pipelines:
        from transformers import pipeline

        print(f"Loading pipeline: {task} (first run may download model)...")
        _pipelines[task] = pipeline(task, **kwargs)
    return _pipelines[task]


def demo_sentiment() -> None:
    records = load_sentiment()
    record = records[0]
    clf = get_pipeline("sentiment-analysis")
    result = clf(record["text"])[0]
    print(f"\n--- Sentiment: {record['id']} ---")
    print(f"Text: {record['text'][:100]}...")
    print(f"Actual: {record['sentiment']}")
    print(f"Pipeline: {result['label']} ({result['score']:.3f})")
    confidence = result["score"]
    if confidence > 0.9:
        print("\nHigh confidence — the model is quite sure about this one.")
    elif confidence > 0.7:
        print("\nModerate confidence — the model leans this way but isn't certain.")
    else:
        print("\nLow confidence (near 0.5) — the model is uncertain. This text")
        print("may be ambiguous or fall outside the model's training domain.")


def demo_ner() -> None:
    print("\nTransformer NER groups adjacent B-I tokens into entity spans,")
    print("similar to spaCy (Module 6) but using contextual embeddings.\n")
    statements = load_statements()
    stmt = statements[0]
    ner = get_pipeline("ner", grouped_entities=True)
    entities = ner(stmt["raw_text"])
    print(f"--- NER: {stmt['id']} ---")
    for ent in entities[:8]:
        print(f"  {ent.get('entity_group', ent.get('entity'))}: {ent['word']}")
    print(f"\n{len(entities)} entities found. Compare with spaCy NER on the")
    print("same text — transformers often catch names spaCy misses.")


def demo_summarize() -> None:
    statements = load_statements()
    long_stmt = max(statements, key=lambda s: len(s["raw_text"]))
    input_len = len(long_stmt["raw_text"])
    print(f"\nInput: {input_len} chars. Target output: 20–60 tokens.")
    print("The model condenses by attending to the most salient tokens —")
    print("not just truncating.\n")
    summarizer = get_pipeline("summarization")
    summary = summarizer(long_stmt["raw_text"], max_length=60, min_length=20, do_sample=False)
    summary_text = summary[0]["summary_text"]
    print(f"--- Summarize: {long_stmt['id']} ({input_len} chars → {len(summary_text)} chars) ---")
    print(summary_text)


def demo_zero_shot() -> None:
    print("\nZero-shot uses natural language inference: for each label, the")
    print("model asks 'does this text entail this label?' The highest")
    print("entailment score wins.\n")
    records = load_sentiment()
    record = records[2]
    clf = get_pipeline("zero-shot-classification")
    result = clf(record["text"], candidate_labels=["calm", "hostile"])
    print(f"--- Zero-shot: {record['id']} ---")
    print(f"Text: {record['text'][:100]}...")
    print(f"Actual: {record['sentiment']}")
    winner = result["labels"][0]
    for label, score in zip(result["labels"], result["scores"], strict=True):
        marker = " ←" if label == winner else ""
        print(f"  {label}: {score:.3f}{marker}")
    print("\nNo training data was used — the model generalises from its")
    print("pre-training. Domain-specific nuance may need fine-tuning.")


def demo_tokenizer() -> None:
    statements = load_statements()
    text = statements[0]["raw_text"][:120]
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    subwords = [enc.decode([t]) for t in tokens[:15]]
    print(f"\n--- tiktoken (first 15 subwords) ---")
    print(f"Text: {text[:80]}...")
    print(f"Tokens ({len(tokens)} total): {subwords}")
    whole_words = sum(1 for t in tokens if enc.decode([t]).strip().isalpha())
    fragments = len(tokens) - whole_words
    print(f"\nFull words: {whole_words}  Sub-word fragments/punctuation: {fragments}")
    print("BPE splits rare words into known sub-parts. Common words stay whole.")
    print("Token count matters for API limits (e.g. 4096 tokens max).")


def demo_context() -> None:
    print("\nSame word, different context. A static embedding (Word2Vec)")
    print("would give identical vectors. A transformer produces different")
    print("representations at inference time.\n")
    sentences = [
        "The accountant left the office at six.",
        "The river bank was muddy after the rain.",
    ]
    word = "bank" if "bank" in sentences[1] else "accountant"
    enc = tiktoken.get_encoding("cl100k_base")
    print(f"--- Token IDs for '{word}' ---")
    for s in sentences:
        tokens = enc.encode(s)
        ids = tokens[:8]
        print(f"  {s:<50} → first tokens: {ids}")
    print("\nNote: tiktoken is a tokenizer — the token IDs are the same")
    print("regardless of context. The contextual difference happens inside")
    print("the transformer model, not the tokenizer.")


def main() -> None:
    while True:
        print("\nInkwell Investigations - Transformer Lab")
        print("=" * 40)
        print("\n1. Sentiment pipeline")
        print("2. NER pipeline")
        print("3. Summarization (long statement)")
        print("4. Zero-shot classification")
        print("5. tiktoken explorer")
        print("6. Context / token IDs")
        print("0. Quit")

        choice = input("\nChoice: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            demo_sentiment()
        elif choice == "2":
            demo_ner()
        elif choice == "3":
            demo_summarize()
        elif choice == "4":
            demo_zero_shot()
        elif choice == "5":
            demo_tokenizer()
        elif choice == "6":
            demo_context()
        else:
            print("Unknown option.")


if __name__ == "__main__":
    main()
