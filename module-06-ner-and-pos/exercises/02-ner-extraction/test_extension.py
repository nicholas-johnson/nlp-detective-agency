"""Tests for Exercise 02 - NER Extraction (Part B extension)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("spacy")

from start import evaluate_ner, load_nlp

CONLL_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "conll_ner_sample.json"
)


@pytest.fixture(scope="module")
def nlp():
    return load_nlp()


class TestConllExtension:
    def test_conll_sample_loads(self):
        samples = json.loads(CONLL_PATH.read_text())
        assert len(samples) >= 35

    def test_conll_metrics_non_empty(self, nlp):
        samples = json.loads(CONLL_PATH.read_text())
        all_pred = []
        all_gold = []
        for record in samples:
            doc = nlp(record["text"])
            for ent in doc.ents:
                label = ent.label_
                if label in ("GPE", "FAC"):
                    label = "LOC"
                all_pred.append({"text": ent.text, "label": label})
            all_gold.extend(record["entities"])
        metrics = evaluate_ner(all_pred, all_gold)
        assert len(metrics) > 0
        assert any(m["support"] > 0 for m in metrics.values())

    def test_conll_person_recall_reasonable(self, nlp):
        samples = json.loads(CONLL_PATH.read_text())
        all_pred = []
        all_gold = []
        for record in samples:
            doc = nlp(record["text"])
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    all_pred.append({"text": ent.text, "label": "PERSON"})
            all_gold.extend(e for e in record["entities"] if e["label"] == "PERSON")
        if not all_gold:
            pytest.skip("No PERSON entities in sample")
        metrics = evaluate_ner(all_pred, all_gold)
        assert metrics["PERSON"]["recall"] > 0.2
