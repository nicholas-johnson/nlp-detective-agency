"""Tests for Exercise 03 - Fine-Tuning (Part A)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import start

SENTIMENT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "inkwell"
    / "witness_sentiment.json"
)


class TestLoadData:
    def test_loads_14_records(self):
        records = start.load_sentiment_data(SENTIMENT_PATH)
        assert len(records) == 14

    def test_split_stratified(self):
        records = start.load_sentiment_data(SENTIMENT_PATH)
        train_t, test_t, train_l, test_l = start.split_data(records)
        assert len(train_t) + len(test_t) == 14
        assert set(train_l) <= {"calm", "hostile"}


class TestSklearnBaseline:
    def test_baseline_metrics(self):
        records = start.load_sentiment_data(SENTIMENT_PATH)
        texts = [r["text"] for r in records]
        labels = [r["sentiment"] for r in records]
        metrics = start.sklearn_baseline(texts, labels)
        assert "accuracy" in metrics
        assert "f1_hostile" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0


class TestPredictSamples:
    def test_returns_predictions(self):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        # Simulate model output
        import torch

        mock_logits = torch.tensor([[0.2, 0.8]])
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_model.return_value = mock_output
        mock_model.eval = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long), "attention_mask": torch.ones(1, 10, dtype=torch.long)}

        results = start.predict_samples(mock_model, mock_tokenizer, ["Test text."])
        assert len(results) == 1
        assert "predicted_label" in results[0]
        assert "confidence" in results[0]
        assert results[0]["predicted_label"] in ("calm", "hostile")


class TestCompareBaseline:
    def test_compare_structure(self):
        cmp = start.compare_to_baseline(
            {"eval_loss": 0.4, "eval_accuracy": 0.85},
            {"accuracy": 0.75, "f1_hostile": 0.7},
        )
        assert cmp["transformer"]["eval_accuracy"] == 0.85
        assert cmp["sklearn_baseline"]["accuracy"] == 0.75
