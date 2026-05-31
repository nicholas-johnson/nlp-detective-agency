"""Tests for Exercise 03 - Fine-Tuning (Part B)."""

import json
from pathlib import Path

import pytest

import start

REVIEWS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "public"
    / "movie_reviews_sample.json"
)


class TestReviewsExtension:
    def test_reviews_load(self):
        reviews = json.loads(REVIEWS_PATH.read_text())
        assert len(reviews) >= 35

    def test_baseline_on_reviews(self):
        reviews = json.loads(REVIEWS_PATH.read_text())
        texts = [r["text"] for r in reviews]
        labels = ["calm" if r["sentiment"] == "pos" else "hostile" for r in reviews]
        metrics = start.sklearn_baseline(texts, labels)
        assert metrics["accuracy"] >= 0.5
