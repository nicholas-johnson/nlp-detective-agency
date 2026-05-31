"""Tests for Exercise 03 - Article Matcher."""

from pathlib import Path

import pytest

from start import (
    article_audit,
    build_tfidf_matrix,
    load_articles,
    same_category_rate,
    top_similar_pairs,
)
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "newsgroups_articles.json"


@pytest.fixture()
def records():
    return load_articles(DATA_PATH)


class TestLoadArticles:
    def test_loads_40_records(self, records):
        assert len(records) == 40

    def test_four_categories(self, records):
        assert len({r["category"] for r in records}) == 4


class TestSimilarity:
    def test_scores_in_range(self, records):
        texts = [r["text"] for r in records]
        matrix, _ = build_tfidf_matrix(texts)
        sim = cosine_similarity(matrix)
        pairs = top_similar_pairs(sim, [r["id"] for r in records], n=5)
        assert all(0.0 <= p["score"] <= 1.0 for p in pairs)

    def test_pairs_sorted_descending(self, records):
        texts = [r["text"] for r in records]
        matrix, _ = build_tfidf_matrix(texts)
        sim = cosine_similarity(matrix)
        pairs = top_similar_pairs(sim, [r["id"] for r in records], n=5)
        scores = [p["score"] for p in pairs]
        assert scores == sorted(scores, reverse=True)

    def test_same_category_rate_range(self, records):
        report = article_audit(records)
        assert 0.0 <= report["same_category_rate"] <= 1.0


class TestAudit:
    def test_report_keys(self, records):
        report = article_audit(records)
        assert report["total_articles"] == 40
        assert "vocab_sizes" in report
        assert len(report["top_pairs"]) == 5
