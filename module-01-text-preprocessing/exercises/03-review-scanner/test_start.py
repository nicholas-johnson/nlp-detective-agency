"""Tests for Exercise 03 - Review Scanner."""

from pathlib import Path

import pytest

from start import (
    class_term_frequencies,
    distinctive_terms,
    load_reviews,
    preprocess_review,
    review_audit,
)

DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "movie_reviews_sample.json"


@pytest.fixture()
def records():
    return load_reviews(DATA_PATH)


class TestLoadReviews:
    def test_loads_40_records(self, records):
        assert len(records) == 40

    def test_balanced_sentiment(self, records):
        pos = sum(1 for r in records if r["sentiment"] == "pos")
        neg = sum(1 for r in records if r["sentiment"] == "neg")
        assert pos == 20
        assert neg == 20


class TestPreprocess:
    def test_reduces_raw_length(self, records):
        raw = len(records[0]["text"].split())
        cleaned = len(preprocess_review(records[0]["text"]))
        assert cleaned < raw

    def test_min_token_length(self, records):
        tokens = preprocess_review(records[0]["text"])
        assert all(len(t) >= 3 for t in tokens)


class TestFrequencies:
    def test_two_classes(self, records):
        freqs = class_term_frequencies(records)
        assert "pos" in freqs
        assert "neg" in freqs
        assert len(freqs["pos"]) > 0
        assert len(freqs["neg"]) > 0

    def test_distinctive_terms(self, records):
        freqs = class_term_frequencies(records)
        terms = distinctive_terms(freqs["pos"], freqs["neg"], n=5)
        assert len(terms) == 5
        assert all(t[1] >= 1.0 for t in terms)


class TestAudit:
    def test_report_keys(self, records):
        report = review_audit(records)
        assert report["total_vocab"] > 0
        assert report["avg_tokens"] > 0
        assert len(report["pos_top"]) <= 10
        assert len(report["distinctive"]) == 10
