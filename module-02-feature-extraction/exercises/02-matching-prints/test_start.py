"""Tests for Exercise 02 - Matching Prints."""

from pathlib import Path

import pytest

from start import (
    build_tfidf_matrix,
    compare_ngram_vocab_sizes,
    distinctive_terms,
    load_statements,
    most_similar_pair,
    similarity_report,
    texts_for_case,
)

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
STATEMENTS_PATH = DATA_DIR / "inkwell" / "statements.json"


@pytest.fixture()
def statements():
    return load_statements(STATEMENTS_PATH)


class TestTfidfMatrix:
    def test_matrix_shape(self, statements):
        _, texts = texts_for_case(statements, "CASE-42")
        matrix, feature_names = build_tfidf_matrix(texts)
        assert matrix.shape == (4, len(feature_names))

    def test_distinctive_terms(self, statements):
        _, texts = texts_for_case(statements, "CASE-42")
        matrix, feature_names = build_tfidf_matrix(texts)
        terms = distinctive_terms(matrix, feature_names, 0, n=3)
        assert len(terms) <= 3
        assert all(isinstance(w, float) for _, w in terms)


class TestMostSimilarPair:
    def test_returns_pair_for_case42(self, statements):
        doc_ids, texts = texts_for_case(statements, "CASE-42")
        matrix, _ = build_tfidf_matrix(texts)
        result = most_similar_pair(matrix, doc_ids)
        assert result is not None
        id_a, id_b, score = result
        assert id_a != id_b
        assert 0.0 <= score <= 1.0

    def test_single_doc_returns_none(self):
        from scipy.sparse import csr_matrix

        matrix = csr_matrix([[1.0, 0.0]])
        assert most_similar_pair(matrix, ["STM-001"]) is None


class TestNgramVocab:
    def test_bigram_vocab_larger(self, statements):
        _, texts = texts_for_case(statements, "CASE-42")
        sizes = compare_ngram_vocab_sizes(texts)
        assert sizes["bigram"] >= sizes["unigram"]


class TestSimilarityReport:
    def test_report_shape(self, statements):
        report = similarity_report(statements, "CASE-42")
        assert report["case_id"] == "CASE-42"
        assert report["most_similar"] is not None
        assert "unigram" in report["ngram_vocab_sizes"]
        assert "bigram" in report["ngram_vocab_sizes"]

    def test_unknown_case(self, statements):
        report = similarity_report(statements, "CASE-99")
        assert report["most_similar"] is None
