"""
Exercise 02 - Tokenization (solution)
"""

import argparse
import json
import statistics
from pathlib import Path

import tiktoken

STATEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
)
SMS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "sms_spam_sample.json"
)

TIKTOKEN_MODEL = "cl100k_base"
HF_MODEL = "distilbert-base-uncased"


def load_tiktoken_encoding(name: str = TIKTOKEN_MODEL):
    return tiktoken.get_encoding(name)


def count_tokens(encoding, text: str) -> int:
    return len(encoding.encode(text))


def show_subwords(encoding, text: str) -> list[str]:
    token_ids = encoding.encode(text)
    return [encoding.decode([tid]) for tid in token_ids]


def batch_token_stats(encoding, texts: list[str]) -> dict:
    counts = [count_tokens(encoding, t) for t in texts]
    return {
        "min": min(counts) if counts else 0,
        "max": max(counts) if counts else 0,
        "mean": round(statistics.mean(counts), 2) if counts else 0.0,
        "total": sum(counts),
    }


def truncate_analysis(encoding, text: str, max_tokens: int) -> dict:
    token_ids = encoding.encode(text)
    truncated = token_ids[:max_tokens]
    return {
        "original_tokens": len(token_ids),
        "truncated_tokens": len(truncated),
        "truncated_text": encoding.decode(truncated),
        "was_truncated": len(token_ids) > max_tokens,
    }


def compare_tokenizers(text: str, tiktoken_encoding=None, hf_model: str = HF_MODEL) -> dict:
    if tiktoken_encoding is None:
        tiktoken_encoding = load_tiktoken_encoding()
    tik_ids = tiktoken_encoding.encode(text)
    tik_tokens = [tiktoken_encoding.decode([i]) for i in tik_ids]

    hf_tokens: list[str] = []
    hf_count = 0
    try:
        from transformers import AutoTokenizer

        hf_tok = AutoTokenizer.from_pretrained(hf_model)
        hf_encoded = hf_tok(text, truncation=False, add_special_tokens=True)
        hf_count = len(hf_encoded["input_ids"])
        hf_tokens = hf_tok.convert_ids_to_tokens(hf_encoded["input_ids"])
    except ImportError:
        hf_tokens = ["(transformers not installed)"]

    return {
        "tiktoken_count": len(tik_ids),
        "tiktoken_tokens": tik_tokens[:20],
        "hf_count": hf_count,
        "hf_tokens": hf_tokens[:20],
    }


def run_inkwell() -> None:
    encoding = load_tiktoken_encoding()
    statements = json.loads(STATEMENTS_PATH.read_text())
    texts = [s["raw_text"] for s in statements]

    print("Inkwell Investigations - Tokenization Lab")
    print("=" * 48)

    stats = batch_token_stats(encoding, texts)
    print(f"\nWitness statements ({len(texts)} docs):")
    print(f"  Tokens - min: {stats['min']}, max: {stats['max']}, mean: {stats['mean']}")

    sample = texts[0]
    subwords = show_subwords(encoding, sample)[:15]
    print(f"\nSubwords (STM-001, first 15): {subwords}")

    trunc = truncate_analysis(encoding, texts[-2], max_tokens=50)
    print(f"\nTruncation at 50 tokens (STM-009):")
    print(f"  Original: {trunc['original_tokens']} → Truncated: {trunc['truncated_tokens']}")

    cmp = compare_tokenizers(sample)
    print(f"\nTokenizer comparison (STM-001 excerpt):")
    print(f"  tiktoken: {cmp['tiktoken_count']} tokens")
    print(f"  HF ({HF_MODEL}): {cmp['hf_count']} tokens")


def run_real_world() -> None:
    encoding = load_tiktoken_encoding()
    statements = json.loads(STATEMENTS_PATH.read_text())
    sms = json.loads(SMS_PATH.read_text())

    inkwell_stats = batch_token_stats(encoding, [s["raw_text"] for s in statements])
    sms_stats = batch_token_stats(encoding, [m["text"] for m in sms])

    print("Tokenization Lab - Domain Comparison")
    print("=" * 48)
    print(f"Inkwell statements - mean tokens: {inkwell_stats['mean']}")
    print(f"SMS messages       - mean tokens: {sms_stats['mean']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-world", action="store_true")
    args = parser.parse_args()
    if args.real_world:
        run_real_world()
    else:
        run_inkwell()


if __name__ == "__main__":
    main()
