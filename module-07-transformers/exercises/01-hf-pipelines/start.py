"""
Exercise 01 - HF Pipelines
Run Hugging Face pipelines for sentiment, NER, and zero-shot classification.
"""

import argparse
import json
from pathlib import Path

SENTIMENT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "inkwell"
    / "witness_sentiment.json"
)
STATEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
)
GOLD_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "inkwell"
    / "statement_entities.json"
)
SMS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "sms_spam_sample.json"
)
CONLL_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "public" / "conll_ner_sample.json"
)

_pipelines: dict = {}


def load_pipeline(task: str, **kwargs):
    """Load and cache a Hugging Face pipeline for the given task."""
    # TODO
    raise NotImplementedError


def load_json(path: Path) -> list[dict]:
    """Load a JSON array from disk."""
    # TODO
    raise NotImplementedError


def classify_sentiment(texts: list[str]) -> list[dict]:
    """Run sentiment-analysis pipeline; return list of {label, score} dicts."""
    # TODO
    raise NotImplementedError


def extract_entities_hf(texts: list[str]) -> list[list[dict]]:
    """Run NER pipeline with grouped_entities=True."""
    # TODO
    raise NotImplementedError


def zero_shot_classify(texts: list[str], labels: list[str]) -> list[dict]:
    """Run zero-shot-classification pipeline with candidate labels."""
    # TODO
    raise NotImplementedError


def compare_ner_to_gold(hf_entities: list[dict], gold_entities: list[dict]) -> dict:
    """Return matched, gold_total, recall comparing HF spans to gold."""
    # TODO
    raise NotImplementedError


def run_inkwell() -> None:
    # TODO
    raise NotImplementedError


def run_real_world() -> None:
    # TODO
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-world", action="store_true")
    args = parser.parse_args()
    if args.real_world:
        run_real_world()
    else:
        run_inkwell()


if __name__ == "__main__":
    main()
