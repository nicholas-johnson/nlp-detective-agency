"""Tests for Exercise 03 - Spam Detector."""

from pathlib import Path

import pytest

from start import (
    build_pipeline,
    compare_classifiers,
    load_messages,
    spam_report,
)

DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "sms_spam_sample.json"


@pytest.fixture()
def records():
    return load_messages(DATA_PATH)


class TestLoadMessages:
    def test_loads_60_records(self, records):
        assert len(records) == 60

    def test_balanced_labels(self, records):
        spam = sum(1 for r in records if r["label"] == "spam")
        ham = sum(1 for r in records if r["label"] == "ham")
        assert spam == 30
        assert ham == 30


class TestBuildPipeline:
    def test_nb_pipeline(self):
        pipeline = build_pipeline("nb")
        assert len(pipeline.steps) == 2

    def test_unknown_classifier(self):
        with pytest.raises(ValueError):
            build_pipeline("unknown")


class TestCompareClassifiers:
    def test_three_classifiers(self, records):
        results = compare_classifiers(records)
        assert len(results) == 3

    def test_f1_in_range(self, records):
        results = compare_classifiers(records)
        for r in results:
            assert 0.0 <= r["f1_mean"] <= 1.0


class TestSpamReport:
    def test_report_keys(self, records):
        report = spam_report(records)
        assert "comparison" in report
        assert "best_classifier" in report
        assert "confusion_matrix" in report
        assert "slipped_through" in report

    def test_confusion_matrix_shape(self, records):
        report = spam_report(records)
        cm = report["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_slipped_through_are_ids(self, records):
        report = spam_report(records)
        for msg_id in report["slipped_through"]:
            assert msg_id.startswith("SMS-")
