"""Tests for Exercise 01 - Transformer Inference Lab (Part A)."""

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
STATEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
)


def _mock_sentiment(texts, **kwargs):
    if isinstance(texts, str):
        texts = [texts]
    return [{"label": "NEGATIVE", "score": 0.9} for _ in texts]


def _mock_ner(text, **kwargs):
    return [{"entity_group": "PER", "word": "Reeves", "score": 0.95, "start": 0, "end": 6}]


def _mock_zero_shot(text, candidate_labels, **kwargs):
    return {"labels": candidate_labels, "scores": [0.8, 0.2]}


def _mock_summarize(text, **kwargs):
    return [{"summary_text": "Two figures argued near the pier."}]


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
        elif task == "summarization":
            pipe.side_effect = _mock_summarize
        return pipe

    with patch.object(start, "load_model", side_effect=factory):
        yield


class TestLoadJson:
    def test_loads_sentiment(self):
        records = start.load_json(SENTIMENT_PATH)
        assert len(records) == 14

    def test_loads_statements(self):
        records = start.load_json(STATEMENTS_PATH)
        assert len(records) == 10


class TestAnalyseSentiment:
    def test_returns_results(self):
        records = start.load_json(SENTIMENT_PATH)[:3]
        results = start.analyse_sentiment(records)
        assert len(results) == 3
        assert "predicted" in results[0]
        assert "score" in results[0]
        assert "actual" in results[0]


class TestExtractEntities:
    def test_returns_entities(self):
        statements = start.load_json(STATEMENTS_PATH)[:2]
        results = start.extract_entities(statements)
        assert len(results) == 2
        assert "entities" in results[0]
        assert results[0]["entities"][0]["word"] == "Reeves"


class TestBuildEvidenceBoard:
    def test_merges_entities_by_case(self):
        entity_results = [
            {"case_id": "CASE-42", "entities": [{"entity_group": "PER", "word": "Reeves"}]},
            {"case_id": "CASE-42", "entities": [{"entity_group": "PER", "word": "Marsh"}]},
            {"case_id": "CASE-17", "entities": [{"entity_group": "PER", "word": "Corbin"}]},
        ]
        board = start.build_evidence_board(entity_results, "CASE-42")
        assert "PER" in board
        assert "Reeves" in board["PER"]
        assert "Marsh" in board["PER"]
        assert "Corbin" not in board.get("PER", set())


class TestClassifyZeroShot:
    def test_returns_predictions(self):
        records = start.load_json(SENTIMENT_PATH)[:2]
        results = start.classify_zero_shot(records, ["calm", "hostile"])
        assert len(results) == 2
        assert "predicted" in results[0]
        assert "scores" in results[0]


class TestSummariseLongest:
    def test_returns_summary(self):
        statements = start.load_json(STATEMENTS_PATH)
        result = start.summarise_longest(statements)
        assert "summary" in result
        assert result["original_words"] > result["summary_words"]
