"""Tests for Exercise 01 - Alias Map."""

from pathlib import Path

import pytest

from start import (
    alias_map,
    load_corpus,
    nearest_neighbours,
    tokenize_corpus,
    train_word2vec,
)

STATEMENTS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
COLD_CASES_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "cold_cases.json"


@pytest.fixture()
def sentences():
    texts = load_corpus(STATEMENTS_PATH, COLD_CASES_PATH)
    return tokenize_corpus(texts)


@pytest.fixture()
def model(sentences):
    return train_word2vec(sentences)


class TestLoadCorpus:
    def test_loads_37_texts(self):
        texts = load_corpus(STATEMENTS_PATH, COLD_CASES_PATH)
        assert len(texts) == 37


class TestTokenize:
    def test_returns_list_of_lists(self, sentences):
        assert isinstance(sentences, list)
        assert all(isinstance(s, list) for s in sentences)
        assert all(isinstance(t, str) for s in sentences for t in s)


class TestWord2Vec:
    def test_vocab_non_empty(self, model):
        assert len(model.wv) > 0

    def test_nearest_neighbours(self, model):
        word = model.wv.index_to_key[0]
        neighbours = nearest_neighbours(model, word, n=3)
        assert len(neighbours) == 3
        assert all(isinstance(s, float) for _, s in neighbours)

    def test_alias_map_keys(self, model):
        seeds = ["dock", "warehouse"]
        mapping = alias_map(model, seeds)
        assert set(mapping.keys()) == set(seeds)
