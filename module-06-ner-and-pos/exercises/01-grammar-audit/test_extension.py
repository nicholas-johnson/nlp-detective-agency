"""Tests for Exercise 01 - Grammar Audit (Part B extension)."""

from pathlib import Path

import pytest

pytest.importorskip("spacy")

from start import extract_svo_triples, load_nlp, load_ud_sample, score_svo

UD_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "ud_ewt_sample.json"


@pytest.fixture(scope="module")
def nlp():
    return load_nlp()


class TestUdExtension:
    def test_loads_ud_sample(self):
        samples = load_ud_sample(UD_PATH)
        assert len(samples) >= 30
        assert "gold_svo" in samples[0]

    def test_ud_recall_above_zero(self, nlp):
        samples = load_ud_sample(UD_PATH)
        total_matched = 0
        total_gold = 0
        for record in samples:
            doc = nlp(record["text"])
            predicted = extract_svo_triples(doc)
            scores = score_svo(predicted, record["gold_svo"])
            total_matched += scores["matched"]
            total_gold += scores["total"]
        recall = total_matched / total_gold if total_gold else 0
        assert recall > 0.3

    def test_score_svo_empty_gold(self):
        result = score_svo([], [])
        assert result["recall"] == 0.0
