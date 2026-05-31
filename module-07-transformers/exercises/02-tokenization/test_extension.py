"""Tests for Exercise 02 - Tokenization (Part B)."""

import json
from pathlib import Path

import pytest

import start

STATEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
)
SMS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "sms_spam_sample.json"
)


@pytest.fixture(scope="module")
def encoding():
    return start.load_tiktoken_encoding()


class TestDomainComparison:
    def test_sms_stats(self, encoding):
        sms = json.loads(SMS_PATH.read_text())
        stats = start.batch_token_stats(encoding, [m["text"] for m in sms])
        assert stats["total"] > 100

    def test_inkwell_vs_sms_mean(self, encoding):
        statements = json.loads(STATEMENTS_PATH.read_text())
        sms = json.loads(SMS_PATH.read_text())
        ink = start.batch_token_stats(encoding, [s["raw_text"] for s in statements])
        sms_s = start.batch_token_stats(encoding, [m["text"] for m in sms])
        assert ink["mean"] > 0
        assert sms_s["mean"] > 0
