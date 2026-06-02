"""Tests for Exercise 01 - Transformer Inference Lab (Part B)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import start

SMS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "sms_spam_sample.json"
)
CONLL_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "conll_ner_sample.json"
)


@pytest.fixture(autouse=True)
def mock_pipelines():
    def factory(task, **kwargs):
        pipe = MagicMock()
        if task == "sentiment-analysis":
            pipe.side_effect = lambda texts, **kw: [
                {"label": "POSITIVE", "score": 0.7}
                for _ in ([texts] if isinstance(texts, str) else texts)
            ]
        elif task == "ner":
            pipe.side_effect = lambda text, **kw: [
                {"entity_group": "ORG", "word": "EU", "score": 0.9, "start": 0, "end": 2}
            ]
        return pipe

    with patch.object(start, "load_model", side_effect=factory):
        yield


class TestRealWorldData:
    def test_sms_sample_loads(self):
        sms = json.loads(SMS_PATH.read_text())
        assert len(sms) >= 50

    def test_conll_sample_loads(self):
        conll = json.loads(CONLL_PATH.read_text())
        assert len(conll) >= 35

    def test_sms_sentiment_runs(self):
        records = [{"id": "sms-1", "text": "Win a free prize!", "witness": "test", "sentiment": "spam"}]
        results = start.analyse_sentiment(records)
        assert len(results) == 1
        assert "predicted" in results[0]
