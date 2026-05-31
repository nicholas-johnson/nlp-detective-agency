"""Tests for Exercise 03 - Real-World Topics."""

from pathlib import Path

import pytest

from start import (
    build_dtm,
    contingency_matrix,
    dominant_topic,
    fit_lda,
    load_articles,
    top_words,
    topic_audit,
    topic_purity,
)

DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "newsgroups_sample.json"


@pytest.fixture()
def records():
    return load_articles(DATA_PATH)


class TestLoadArticles:
    def test_loads_40_records(self, records):
        assert len(records) == 40

    def test_four_categories(self, records):
        cats = {r["category"] for r in records}
        assert len(cats) == 4


class TestPipeline:
    def test_dtm_shape(self, records):
        texts = [r["text"] for r in records]
        dtm, features = build_dtm(texts)
        assert dtm.shape[0] == 40
        assert len(features) > 0

    def test_lda_four_topics(self, records):
        texts = [r["text"] for r in records]
        dtm, features = build_dtm(texts)
        lda = fit_lda(dtm, n_topics=4)
        words = top_words(lda, features)
        assert len(words) == 4
        assert all(len(w) == 8 for w in words)

    def test_dominant_topic_range(self, records):
        texts = [r["text"] for r in records]
        dtm, _ = build_dtm(texts)
        lda = fit_lda(dtm, n_topics=4)
        assignments = dominant_topic(lda, dtm)
        assert len(assignments) == 40
        assert all(0 <= t < 4 for t in assignments)


class TestPurity:
    def test_perfect_purity(self):
        assignments = [0, 0, 1, 1]
        labels = ["A", "A", "B", "B"]
        result = topic_purity(assignments, labels)
        assert all(p["purity"] == 1.0 for p in result)

    def test_impure_topic(self):
        assignments = [0, 0, 0]
        labels = ["A", "A", "B"]
        result = topic_purity(assignments, labels)
        assert result[0]["purity"] == pytest.approx(2 / 3, abs=0.01)
        assert result[0]["majority_category"] == "A"


class TestContingency:
    def test_matrix_shape(self):
        assignments = [0, 0, 1, 1]
        labels = ["A", "B", "A", "B"]
        cm = contingency_matrix(assignments, labels, n_topics=2)
        assert len(cm["categories"]) == 2
        assert len(cm["matrix"]) == 2
        assert all(len(row) == 2 for row in cm["matrix"])

    def test_sums_match(self):
        assignments = [0, 0, 1, 1, 0]
        labels = ["A", "A", "B", "B", "B"]
        cm = contingency_matrix(assignments, labels, n_topics=2)
        total = sum(sum(row) for row in cm["matrix"])
        assert total == 5


class TestAudit:
    def test_report_keys(self, records):
        report = topic_audit(records)
        assert "topics" in report
        assert "purity" in report
        assert "avg_purity" in report
        assert "contingency" in report
        assert "assignments" in report

    def test_avg_purity_range(self, records):
        report = topic_audit(records)
        assert 0.0 <= report["avg_purity"] <= 1.0

    def test_assignments_cover_all(self, records):
        report = topic_audit(records)
        assert len(report["assignments"]) == len(records)
        ids = {a["id"] for a in report["assignments"]}
        assert ids == {r["id"] for r in records}
