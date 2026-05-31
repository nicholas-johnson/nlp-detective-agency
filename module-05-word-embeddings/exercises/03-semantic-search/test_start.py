"""Tests for Exercise 03 - Semantic Search."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import start

DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"


def _mock_embeddings(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """Deterministic fake embeddings based on text hash."""
    vectors = []
    for text in texts:
        seed = sum(ord(c) for c in text) % 1000
        rng = np.random.default_rng(seed)
        vec = rng.random(32).tolist()
        vectors.append(vec)
    return vectors


@pytest.fixture()
def records():
    return start.load_statements(DATA_PATH)


@pytest.fixture(autouse=True)
def mock_embed():
    with patch.object(start, "embed_texts", side_effect=_mock_embeddings):
        yield


class TestLoadStatements:
    def test_loads_records(self, records):
        assert len(records) == 10
        assert all("raw_text" in r for r in records)


class TestCosinePairs:
    def test_scores_in_range(self):
        ids = ["A", "B", "C"]
        embeddings = [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]
        pairs = start.cosine_pairs(embeddings, ids, n=2)
        assert len(pairs) == 2
        assert all(0.0 <= p["score"] <= 1.0 for p in pairs)

    def test_sorted_descending(self):
        ids = ["A", "B", "C", "D"]
        embeddings = np.random.default_rng(42).random((4, 8)).tolist()
        pairs = start.cosine_pairs(embeddings, ids, n=3)
        scores = [p["score"] for p in pairs]
        assert scores == sorted(scores, reverse=True)


class TestCompare:
    def test_both_keys(self, records):
        result = start.compare_with_tfidf(records, n=3)
        assert "embedding_pairs" in result
        assert "tfidf_pairs" in result
        assert len(result["embedding_pairs"]) == 3


class TestSearchReport:
    def test_report_keys(self, records):
        report = start.search_report(records)
        assert "embedding_pairs" in report
        assert "tfidf_pairs" in report
        assert "only_in_embeddings" in report
