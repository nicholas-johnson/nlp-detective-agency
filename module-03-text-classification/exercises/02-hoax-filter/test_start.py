"""Tests for Exercise 02 — Hoax Filter."""

from pathlib import Path

import pytest

from start import (
    build_pipeline,
    compare_classifiers,
    confusion,
    hoax_report,
    load_tips,
)

DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "tips.json"


@pytest.fixture()
def records():
    return load_tips(DATA_PATH)


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
        names = {r["name"] for r in results}
        assert "Naive Bayes" in names
        assert "Logistic Regression" in names
        assert "Linear SVM" in names

    def test_f1_in_range(self, records):
        results = compare_classifiers(records)
        for r in results:
            assert 0.0 <= r["f1_mean"] <= 1.0
            assert r["f1_std"] >= 0.0

    def test_sorted_descending(self, records):
        results = compare_classifiers(records)
        means = [r["f1_mean"] for r in results]
        assert means == sorted(means, reverse=True)


class TestConfusion:
    def test_shape(self):
        cm = confusion(
            ["credible", "credible", "hoax", "hoax"],
            ["credible", "hoax", "hoax", "hoax"],
        )
        assert cm == [[1, 1], [0, 2]]


class TestHoaxReport:
    def test_report_keys(self, records):
        report = hoax_report(records)
        assert "comparison" in report
        assert "best_classifier" in report
        assert "confusion_matrix" in report
        assert "slipped_through" in report

    def test_confusion_matrix_shape(self, records):
        report = hoax_report(records)
        cm = report["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_slipped_through_are_ids(self, records):
        report = hoax_report(records)
        for tip_id in report["slipped_through"]:
            assert tip_id.startswith("TIP-")
