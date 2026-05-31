"""Tests for Exercise 02 - Embedding Compass."""

import pytest

from start import analogy, compass_report, load_pretrained, odd_one_out, similarity


@pytest.fixture(scope="module")
def glove_model():
    return load_pretrained()


class TestSimilarity:
    def test_score_in_range(self, glove_model):
        score = similarity(glove_model, "king", "queen")
        assert -1.0 <= score <= 1.0

    def test_related_words_score_higher(self, glove_model):
        related = similarity(glove_model, "king", "queen")
        unrelated = similarity(glove_model, "king", "car")
        assert related > unrelated


class TestAnalogy:
    def test_returns_string(self, glove_model):
        answer = analogy(glove_model, ["king", "woman"], ["man"])
        assert isinstance(answer, str)
        assert len(answer) > 0


class TestOddOneOut:
    def test_returns_one_of_input(self, glove_model):
        words = ["dock", "pier", "ledger", "warehouse"]
        answer = odd_one_out(glove_model, words)
        assert answer in words


class TestCompassReport:
    def test_report_keys(self, glove_model):
        report = compass_report(glove_model)
        assert "similarities" in report
        assert "analogies" in report
        assert "odd_one_out" in report
        assert len(report["similarities"]) == 3
