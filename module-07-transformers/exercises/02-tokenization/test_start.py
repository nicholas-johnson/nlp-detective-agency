"""Tests for Exercise 02 - Text Generation (Part A)."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("transformers")

import start


def _mock_generate(text, **kwargs):
    n = kwargs.get("num_return_sequences", 1)
    return [{"generated_text": text + " The fog rolled in."} for _ in range(n)]


@pytest.fixture(autouse=True)
def mock_generator():
    pipe = MagicMock(side_effect=_mock_generate)
    with patch.object(start, "load_generator", return_value=pipe):
        yield pipe


class TestLoadJson:
    def test_loads_statements(self):
        records = start.load_json(start.STATEMENTS_PATH)
        assert len(records) == 10


class TestContinueStatement:
    def test_returns_continuation(self):
        result = start.continue_statement("I saw him near the docks.")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_does_not_include_prompt(self):
        prompt = "I saw him near the docks."
        result = start.continue_statement(prompt)
        assert not result.startswith(prompt)


class TestGenerateVariants:
    def test_returns_n_variants(self):
        results = start.generate_variants("The docks were empty.", n=3, temperature=0.7)
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)


class TestInterrogationPrompt:
    def test_returns_question(self):
        result = start.interrogation_prompt("Eleanor Marsh", "I was at the warehouse.")
        assert isinstance(result, str)
        assert len(result) > 0


class TestBatchGenerate:
    def test_returns_results_for_all(self):
        statements = start.load_json(start.STATEMENTS_PATH)[:3]
        results = start.batch_generate(statements)
        assert len(results) == 3
        assert "continuation" in results[0]
        assert "id" in results[0]
