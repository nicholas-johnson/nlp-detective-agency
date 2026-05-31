"""
Exercise 02 - Tokenization
Explore BPE tokenization with tiktoken; compare to Hugging Face AutoTokenizer.
"""

import argparse
import json
from pathlib import Path

STATEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
)
SMS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "sms_spam_sample.json"
)

TIKTOKEN_MODEL = "cl100k_base"
HF_MODEL = "distilbert-base-uncased"


def load_tiktoken_encoding(name: str = TIKTOKEN_MODEL):
    """Return a tiktoken encoding (default cl100k_base)."""
    # TODO
    raise NotImplementedError


def count_tokens(encoding, text: str) -> int:
    """Return token count for text."""
    # TODO
    raise NotImplementedError


def show_subwords(encoding, text: str) -> list[str]:
    """Return list of subword strings after BPE encode/decode per token."""
    # TODO
    raise NotImplementedError


def batch_token_stats(encoding, texts: list[str]) -> dict:
    """Return min, max, mean, total token counts across texts."""
    # TODO
    raise NotImplementedError


def truncate_analysis(encoding, text: str, max_tokens: int) -> dict:
    """Truncate to max_tokens; return original/truncated counts and text."""
    # TODO
    raise NotImplementedError


def compare_tokenizers(text: str, tiktoken_encoding=None, hf_model: str = HF_MODEL) -> dict:
    """Side-by-side tiktoken vs AutoTokenizer token counts and sample tokens."""
    # TODO
    raise NotImplementedError


def run_inkwell() -> None:
    # TODO
    raise NotImplementedError


def run_real_world() -> None:
    # TODO
    raise NotImplementedError


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
