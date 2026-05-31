"""Tests for Capstone FastAPI service."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import start
from api import create_app

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


@pytest.fixture
def client(tmp_path):
    records = start.load_dataset("movie_reviews")
    train, test = start.split_records(records)
    pipeline = start.train_baseline(train, "lr")
    metrics = start.evaluate(
        pipeline,
        [start.preprocess_text(r["text"]) for r in test],
        [r["label"] for r in test],
    )
    labels = sorted({r["label"] for r in records})
    out = start.save_artifacts(pipeline, "api_test", labels, metrics, "lr", tmp_path)
    app = create_app(out)
    return TestClient(app)


class TestHealth:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["dataset"] == "api_test"
        assert "pos" in data["labels"]


class TestPredict:
    def test_predict_returns_label(self, client):
        response = client.post(
            "/predict",
            json={"text": "A wonderful and inspiring film."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["label"] in ("pos", "neg")
        assert 0 <= data["confidence"] <= 1
        assert data["model"].startswith("tfidf-")

    def test_predict_logs_request(self, client, tmp_path):
        response = client.post("/predict", json={"text": "Terrible acting."})
        assert response.status_code == 200
        log_path = tmp_path / "api_test" / "predictions.log"
        assert log_path.exists()
        assert "Terrible acting" in log_path.read_text()
