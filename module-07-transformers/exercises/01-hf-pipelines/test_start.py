"""Tests for Exercise 01 - HF Pipelines (Part A)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("transformers")

import start

SENTIMENT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "inkwell"
    / "witness_sentiment.json"
)


def _mock_sentiment(texts, **kwargs):
    if isinstance(texts, str):
        texts = [texts]
    return [{"label": "NEGATIVE", "score": 0.9} for _ in texts]


def _mock_ner(text, **kwargs):
    return [{"entity_group": "PER", "word": "Reeves", "score": 0.95}]


def _mock_zero_shot(texts, candidate_labels, **kwargs):
    if isinstance(texts, str):
        texts = [texts]
    return [
        {"labels": candidate_labels, "scores": [0.8, 0.2]}
        for _ in texts
    ]


@pytest.fixture(autouse=True)
def mock_pipelines():
    def factory(task, **kwargs):
        pipe = MagicMock()
        if task == "sentiment-analysis":
            pipe.side_effect = _mock_sentiment
        elif task == "ner":
            pipe.side_effect = _mock_ner
        elif task == "zero-shot-classification":
            pipe.side_effect = _mock_zero_shot
        return pipe

    with patch.object(start, "load_pipeline", side_effect=factory):
        yield


class TestLoadJson:
    def test_loads_sentiment(self):
        records = start.load_json(SENTIMENT_PATH)
        assert len(records) == 14


class TestClassifySentiment:
    def test_returns_labels(self):
        results = start.classify_sentiment(["Bad day.", "Good day."])
        assert len(results) == 2
        assert "label" in results[0]
        assert "score" in results[0]


class TestExtractEntities:
    def test_returns_entities(self):
        results = start.extract_entities_hf(["Reeves near the docks."])
        assert len(results) == 1
        assert results[0][0]["word"] == "Reeves"


class TestZeroShot:
    def test_returns_labels_and_scores(self):
        results = start.zero_shot_classify(["I am angry."], ["calm", "hostile"])
        assert results[0]["labels"] == ["calm", "hostile"]


class TestCompareNer:
    def test_recall_perfect(self):
        hf = [{"word": "Reeves", "entity_group": "PER"}]
        gold = [{"text": "Reeves", "label": "PERSON"}]
        result = start.compare_ner_to_gold(hf, gold)
        assert result["recall"] == 1.0
