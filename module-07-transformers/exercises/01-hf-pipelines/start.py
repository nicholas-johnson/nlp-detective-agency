"""
Exercise 01 - Transformer Inference Lab
Run local transformer models across four NLP tasks on Inkwell data.
"""

import argparse
import json
import time
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


def load_model(task: str, **kwargs):
    """Load and cache a Hugging Face pipeline. Print model info on first load."""
    # TODO: use transformers.pipeline, cache in _pipelines, print task name on load
    raise NotImplementedError


def load_json(path: Path) -> list[dict]:
    """Load a JSON array from disk."""
    # TODO
    raise NotImplementedError


def analyse_sentiment(records: list[dict]) -> list[dict]:
    """Run sentiment-analysis on witness_sentiment records.

    Return list of {id, witness, predicted, score, actual}.
    """
    # TODO: load pipeline, run on records, map results
    raise NotImplementedError


def extract_entities(statements: list[dict]) -> list[dict]:
    """Run NER pipeline with grouped_entities=True on raw statements.

    Return list of {id, witness, entities: [{entity_group, word}]}.
    """
    # TODO
    raise NotImplementedError


def build_evidence_board(entity_results: list[dict], case_id: str) -> dict:
    """Merge entities from all statements for a given case into sets by type.

    Return {PERSON: set(...), LOC: set(...), DATE: set(...)}.
    """
    # TODO: filter by case_id, merge entity words by group
    raise NotImplementedError


def classify_zero_shot(records: list[dict], labels: list[str]) -> list[dict]:
    """Run zero-shot-classification on witness_sentiment records.

    Return list of {id, predicted, scores, actual}.
    """
    # TODO
    raise NotImplementedError


def summarise_longest(statements: list[dict]) -> dict:
    """Summarise the longest statement.

    Return {id, original_words, summary, summary_words}.
    """
    # TODO: find longest by word count, run summarization pipeline
    raise NotImplementedError


def run_all() -> None:
    """Run all four tasks with timing and formatted output."""
    # TODO: load data, call each function, time each, print results
    raise NotImplementedError


def run_real_world() -> None:
    """Run sentiment on SMS spam + NER on CoNLL sample."""
    # TODO
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-world", action="store_true")
    args = parser.parse_args()
    if args.real_world:
        run_real_world()
    else:
        run_all()


if __name__ == "__main__":
    main()
