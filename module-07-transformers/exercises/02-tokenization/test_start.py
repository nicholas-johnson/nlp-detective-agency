"""Tests for Exercise 02 - Tokenization (Part A)."""

import json
from pathlib import Path

import pytest

import start

STATEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
)


@pytest.fixture(scope="module")
def encoding():
    return start.load_tiktoken_encoding()


class TestTiktoken:
    def test_count_tokens_positive(self, encoding):
        assert start.count_tokens(encoding, "Hello world") > 0

    def test_round_trip_subwords(self, encoding):
        text = "Reeves near the docks"
        subwords = start.show_subwords(encoding, text)
        assert len(subwords) >= 3
        joined = "".join(subwords)
        assert "Reeves" in joined or "reeves" in joined.lower()

    def test_batch_stats(self, encoding):
        statements = json.loads(STATEMENTS_PATH.read_text())
        texts = [s["raw_text"] for s in statements]
        stats = start.batch_token_stats(encoding, texts)
        assert stats["min"] > 0
        assert stats["max"] >= stats["min"]
        assert stats["mean"] > 0

    def test_truncation(self, encoding):
        long_text = "word " * 200
        result = start.truncate_analysis(encoding, long_text, max_tokens=10)
        assert result["was_truncated"]
        assert result["truncated_tokens"] == 10

    def test_compare_tokenizers_keys(self, encoding):
        cmp = start.compare_tokenizers("I saw Reeves.", encoding)
        assert "tiktoken_count" in cmp
        assert "hf_count" in cmp
        assert cmp["tiktoken_count"] > 0
