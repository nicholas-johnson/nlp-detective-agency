"""Tests for Capstone - Open Your Case (Part A)."""

from pathlib import Path
from unittest.mock import patch

import pytest

import start

SMS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "sms_spam_sample.json"
)
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


@pytest.fixture(autouse=True)
def mock_zero_shot():
    def fake_predict(texts, labels):
        if isinstance(texts, str):
            texts = [texts]
        return [
            {"labels": labels, "scores": [0.6] + [0.4 / max(len(labels) - 1, 1)] * (len(labels) - 1)}
            for _ in texts
        ]

    with patch.object(start, "zero_shot_predict", side_effect=fake_predict):
        yield


class TestListDatasets:
    def test_returns_all_keys(self):
        keys = start.list_datasets()
        assert "sms_spam" in keys
        assert "movie_reviews" in keys
        assert "custom" in keys
        assert len(keys) == 6


class TestLoadDataset:
    def test_movie_reviews_schema(self):
        records = start.load_dataset("movie_reviews")
        assert len(records) == 40
        assert all({"id", "text", "label"} <= set(r.keys()) for r in records)

    def test_custom_json(self):
        records = start.load_dataset("custom", path=SMS_PATH)
        assert len(records) == 60
        assert records[0]["label"] in ("ham", "spam")


class TestExplore:
    def test_explore_stats(self):
        records = start.load_dataset("movie_reviews")
        stats = start.explore_dataset(records)
        assert stats["count"] == 40
        assert "pos" in stats["classes"]
        assert stats["avg_length"] > 0


class TestSplit:
    def test_stratified_split(self):
        records = start.load_dataset("movie_reviews")
        train, test = start.split_records(records)
        assert len(train) + len(test) == len(records)
        assert len(test) >= 1


class TestBaseline:
    def test_train_and_evaluate(self):
        records = start.load_dataset("movie_reviews")
        train, test = start.split_records(records)
        pipeline = start.train_baseline(train, "lr")
        metrics = start.evaluate(
            pipeline,
            [start.preprocess_text(r["text"]) for r in test],
            [r["label"] for r in test],
        )
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["f1_macro"] <= 1
        assert "predictions" in metrics
        assert len(metrics["predictions"]) == len(test)


class TestCompare:
    def test_compare_structure(self):
        records = start.load_dataset("movie_reviews")
        train, test = start.split_records(records)
        labels = sorted({r["label"] for r in records})
        pipeline = start.train_baseline(train, "lr")
        test_texts = [start.preprocess_text(r["text"]) for r in test]
        test_labels = [r["label"] for r in test]
        baseline = start.evaluate(pipeline, test_texts, test_labels)
        transformer = start.zero_shot_evaluate(test, labels)
        comparison = start.compare_models(baseline, transformer)
        assert "winner" in comparison
        assert comparison["winner"] in ("baseline", "transformer")


class TestErrorAnalysis:
    def test_finds_misclassifications(self):
        records = [
            {"id": "1", "text": "great film", "label": "pos"},
            {"id": "2", "text": "awful film", "label": "neg"},
        ]
        errors = start.error_analysis(records, ["pos", "pos"])
        assert len(errors) == 1
        assert errors[0]["actual"] == "neg"
        assert errors[0]["predicted"] == "pos"


class TestArtifacts:
    def test_save_and_load(self, tmp_path):
        records = start.load_dataset("movie_reviews")
        train, test = start.split_records(records)
        pipeline = start.train_baseline(train, "lr")
        metrics = start.evaluate(
            pipeline,
            [start.preprocess_text(r["text"]) for r in test],
            [r["label"] for r in test],
        )
        labels = sorted({r["label"] for r in records})
        out = start.save_artifacts(pipeline, "test_run", labels, metrics, "lr", tmp_path)
        loaded, config = start.load_artifacts("test_run", tmp_path)
        assert (out / "baseline.joblib").exists()
        assert config["dataset"] == "test_run"
        assert loaded.predict(["sample text"]) is not None
