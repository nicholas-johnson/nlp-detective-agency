"""Tests for Exercise 01 — Sentiment Triage."""

from pathlib import Path

import pytest

from start import (
    build_sentiment_pipeline,
    load_sentiment_data,
    split_data,
    train_and_evaluate,
    triage_report,
)

DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "witness_sentiment.json"


@pytest.fixture()
def records():
    return load_sentiment_data(DATA_PATH)


class TestLoadAndSplit:
    def test_loads_records(self, records):
        assert len(records) == 14
        assert all("sentiment" in r for r in records)

    def test_split_sizes(self, records):
        X_train, X_test, y_train, y_test = split_data(records)
        assert len(X_train) + len(X_test) == len(records)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)


class TestPipeline:
    def test_pipeline_steps(self):
        pipeline = build_sentiment_pipeline()
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "tfidf"
        assert pipeline.steps[1][0] == "clf"


class TestTrainAndEvaluate:
    def test_metrics_range(self, records):
        result = train_and_evaluate(records)
        assert 0.0 <= result["accuracy"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0

    def test_accuracy_above_chance(self, records):
        result = train_and_evaluate(records)
        assert result["accuracy"] > 0.5

    def test_predictions_shape(self, records):
        result = train_and_evaluate(records)
        assert len(result["predictions"]) == 4  # 25% of 14
        for p in result["predictions"]:
            assert p["actual"] in ("calm", "hostile")
            assert p["predicted"] in ("calm", "hostile")

    def test_triage_report(self, records):
        assert triage_report(records) == train_and_evaluate(records)
