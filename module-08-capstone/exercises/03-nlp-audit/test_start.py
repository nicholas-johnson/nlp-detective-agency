"""Tests for Capstone Exercise 03 - NLP Audit Dashboard."""

from unittest.mock import MagicMock, patch

import pytest

import start


@pytest.fixture()
def inkwell_records():
    return start.load_corpus("inkwell")


@pytest.fixture()
def inkwell_texts(inkwell_records):
    return [r["text"] for r in inkwell_records]


class TestLoadCorpus:
    def test_inkwell_schema(self):
        records = start.load_corpus("inkwell")
        assert len(records) == 10
        assert all({"id", "text", "label"} <= set(r.keys()) for r in records)

    def test_inkwell_labels_are_case_ids(self):
        records = start.load_corpus("inkwell")
        labels = {r["label"] for r in records}
        assert "CASE-42" in labels

    def test_movie_reviews(self):
        records = start.load_corpus("movie_reviews")
        assert len(records) == 40
        assert all("text" in r for r in records)

    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError):
            start.load_corpus("nonexistent")


class TestCorpusStats:
    def test_returns_expected_keys(self, inkwell_records):
        stats = start.corpus_stats(inkwell_records)
        assert "count" in stats
        assert "avg_length" in stats
        assert "vocab_size" in stats
        assert "label_balance" in stats

    def test_count_matches(self, inkwell_records):
        stats = start.corpus_stats(inkwell_records)
        assert stats["count"] == len(inkwell_records)

    def test_vocab_positive(self, inkwell_records):
        stats = start.corpus_stats(inkwell_records)
        assert stats["vocab_size"] > 0


class TestDiscoverTopics:
    def test_returns_topics(self, inkwell_texts):
        topics = start.discover_topics(inkwell_texts, n_topics=2, n_terms=3)
        assert len(topics) == 2
        assert all("topic_id" in t and "top_terms" in t for t in topics)

    def test_top_terms_are_strings(self, inkwell_texts):
        topics = start.discover_topics(inkwell_texts, n_topics=2, n_terms=3)
        for t in topics:
            assert all(isinstance(term, str) for term in t["top_terms"])


class TestSentimentScan:
    def test_returns_distribution(self):
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"label": "POSITIVE", "score": 0.9},
            {"label": "NEGATIVE", "score": 0.8},
            {"label": "POSITIVE", "score": 0.7},
        ]
        with patch.object(start, "_get_sentiment", return_value=mock_pipeline):
            result = start.sentiment_scan(["a", "b", "c"])
        assert "distribution" in result
        assert "scores" in result
        assert result["distribution"]["POSITIVE"] == 2
        assert result["distribution"]["NEGATIVE"] == 1
        assert len(result["scores"]) == 3


class TestEntityCensus:
    def test_finds_entities(self, inkwell_texts):
        census = start.entity_census(inkwell_texts)
        assert isinstance(census, dict)
        all_entity_count = sum(d["count"] for d in census.values())
        assert all_entity_count >= 1

    def test_entity_structure(self, inkwell_texts):
        census = start.entity_census(inkwell_texts)
        for label, data in census.items():
            assert "entities" in data
            assert "count" in data
            assert isinstance(data["entities"], list)


class TestClassificationProbe:
    def test_returns_metrics_with_labels(self):
        records = start.load_corpus("movie_reviews")
        result = start.classification_probe(records)
        assert result is not None
        assert "accuracy" in result
        assert "f1_macro" in result
        assert 0 <= result["accuracy"] <= 1

    def test_returns_none_without_labels(self):
        records = [{"id": "1", "text": "hello", "label": ""}]
        result = start.classification_probe(records)
        assert result is None


class TestBuildReport:
    def test_report_structure(self):
        report = start.build_report(
            dataset="test",
            stats={"count": 5, "avg_length": 100, "vocab_size": 50,
                   "min_length": 10, "max_length": 200, "label_balance": {}},
            topics=[{"topic_id": 0, "top_terms": ["a", "b"]}],
            sentiment={"distribution": {"POSITIVE": 3}, "scores": [0.9]},
            entities={"PERSON": {"entities": ["Alice"], "count": 1}},
            classification=None,
        )
        assert report["dataset"] == "test"
        assert "corpus_stats" in report
        assert "topics" in report
        assert "sentiment" in report
        assert "entity_census" in report
        assert "classification" in report
