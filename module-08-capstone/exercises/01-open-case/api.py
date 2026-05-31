"""
FastAPI inference service for the capstone baseline model.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model: str


def _preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _log_prediction(log_path: Path, text: str, label: str, confidence: float) -> None:
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "text": text[:500],
        "prediction": label,
        "confidence": confidence,
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def create_app(artifacts_dir: Path) -> FastAPI:
    pipeline = joblib.load(artifacts_dir / "baseline.joblib")
    config = json.loads((artifacts_dir / "config.json").read_text())
    log_path = artifacts_dir / "predictions.log"
    classifier = config.get("classifier", "lr")
    model_name = f"tfidf-{classifier}"

    app = FastAPI(title="Inkwell Text Classifier", version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "dataset": config.get("dataset"),
            "labels": config.get("labels", []),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest) -> PredictResponse:
        text = _preprocess(body.text)
        label = pipeline.predict([text])[0]
        confidence = 1.0
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba([text])[0]
            clf = pipeline.named_steps["clf"]
            idx = list(clf.classes_).index(label)
            confidence = round(float(proba[idx]), 3)

        _log_prediction(log_path, body.text, label, confidence)
        return PredictResponse(label=label, confidence=confidence, model=model_name)

    return app
