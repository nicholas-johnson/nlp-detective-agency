"""Tests for Exercise 01 - Archive Themes."""

from pathlib import Path

import pytest

from start import (
    archive_report,
    build_dtm,
    dominant_topics,
    fit_lda,
    load_archive,
    top_words,
)

DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "cold_cases.json"


@pytest.fixture()
def records():
    return load_archive(DATA_PATH)


class TestLoadArchive:
    def test_loads_records(self, records):
        assert len(records) >= 24
        assert all("summary" in r for r in records)


class TestBuildDtm:
    def test_vocab_non_empty(self, records):
        texts = [r["summary"] for r in records]
        dtm, feature_names = build_dtm(texts)
        assert dtm.shape[0] == len(texts)
        assert len(feature_names) > 0


class TestLda:
    def test_four_topics(self, records):
        texts = [r["summary"] for r in records]
        dtm, feature_names = build_dtm(texts)
        lda = fit_lda(dtm, n_topics=4)
        topics = top_words(lda, feature_names)
        assert len(topics) == 4
        assert all(len(t["words"]) == 8 for t in topics)

    def test_dominant_topic_in_range(self, records):
        texts = [r["summary"] for r in records]
        dtm, _ = build_dtm(texts)
        lda = fit_lda(dtm, n_topics=4)
        assignments = dominant_topics(lda, dtm)
        assert len(assignments) == len(records)
        for topic_id, weight in assignments:
            assert 0 <= topic_id < 4
            assert 0.0 <= weight <= 1.0

    def test_reproducible_top_words(self, records):
        texts = [r["summary"] for r in records]
        dtm, feature_names = build_dtm(texts)
        lda1 = fit_lda(dtm, n_topics=4)
        lda2 = fit_lda(dtm, n_topics=4)
        words1 = top_words(lda1, feature_names)
        words2 = top_words(lda2, feature_names)
        assert words1[0]["words"][0][0] == words2[0]["words"][0][0]


class TestArchiveReport:
    def test_report_shape(self, records):
        report = archive_report(records)
        assert len(report["topics"]) == 4
        assert len(report["cases"]) == len(records)
        assert all("dominant_topic" in c for c in report["cases"])
