"""Shared pytest fixtures for Inkwell Investigations NLP exercises."""

import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"
INKWELL_DIR = DATA_DIR / "inkwell"
PUBLIC_DIR = DATA_DIR / "public"


@pytest.fixture()
def statements():
    return json.loads((INKWELL_DIR / "statements.json").read_text())


@pytest.fixture()
def tips():
    return json.loads((INKWELL_DIR / "tips.json").read_text())


@pytest.fixture()
def cold_cases():
    return json.loads((INKWELL_DIR / "cold_cases.json").read_text())


@pytest.fixture()
def witness_sentiment():
    return json.loads((INKWELL_DIR / "witness_sentiment.json").read_text())


@pytest.fixture()
def statement_entities():
    return json.loads((INKWELL_DIR / "statement_entities.json").read_text())
