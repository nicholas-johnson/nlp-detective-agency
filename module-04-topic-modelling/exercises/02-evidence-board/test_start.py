"""Tests for Exercise 02 - Evidence Board."""

from pathlib import Path

import numpy as np
import pytest

from start import (
    compare_models,
    evidence_board,
    fit_nmf,
    label_topics,
    load_archive,
    perplexity_scores,
)

DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "cold_cases.json"


@pytest.fixture()
def records():
    return load_archive(DATA_PATH)


@pytest.fixture()
def dtm_and_tfidf(records):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    texts = [r["summary"] for r in records]
    count_vec = CountVectorizer(stop_words="english", min_df=2)
    dtm = count_vec.fit_transform(texts)
    tfidf_vec = TfidfVectorizer(stop_words="english", min_df=2)
    tfidf = tfidf_vec.fit_transform(texts)
    return dtm, tfidf, count_vec.get_feature_names_out()


class TestPerplexity:
    def test_one_entry_per_k(self, dtm_and_tfidf):
        dtm, _, _ = dtm_and_tfidf
        scores = perplexity_scores(dtm, [3, 4, 5])
        assert len(scores) == 3
        assert scores[0]["k"] == 3


class TestNmf:
    def test_components_non_negative(self, dtm_and_tfidf):
        _, tfidf, _ = dtm_and_tfidf
        nmf = fit_nmf(tfidf, n_topics=4)
        assert np.all(nmf.components_ >= 0)


class TestCompareModels:
    def test_has_lda_and_nmf_keys(self, dtm_and_tfidf):
        from sklearn.decomposition import LatentDirichletAllocation

        dtm, tfidf, features = dtm_and_tfidf
        lda = LatentDirichletAllocation(n_components=4, random_state=42, max_iter=20)
        lda.fit(dtm)
        nmf = fit_nmf(tfidf, 4)
        result = compare_models(lda, nmf, features)
        assert "lda" in result
        assert "nmf" in result
        assert len(result["lda"]) == 4
        assert len(result["nmf"]) == 4


class TestLabelTopics:
    def test_returns_string_labels(self):
        top_words = [
            {"topic_id": 0, "words": [("dock", 1.0), ("pier", 0.9)]},
            {"topic_id": 1, "words": [("ledger", 1.0), ("accountant", 0.8)]},
        ]
        labels = label_topics(top_words)
        assert labels[0] == "Waterfront"
        assert labels[1] == "Financial"


class TestEvidenceBoard:
    def test_board_covers_all_cases(self, records):
        report = evidence_board(records)
        all_ids = {cid for group in report["board"] for cid in group["case_ids"]}
        assert all_ids == {r["id"] for r in records}

    def test_best_k_in_range(self, records):
        report = evidence_board(records, k_values=[3, 4, 5, 6])
        assert report["best_k"] in {3, 4, 5, 6}

    def test_report_keys(self, records):
        report = evidence_board(records)
        assert "lda_topics" in report
        assert "nmf_topics" in report
        assert "board" in report
