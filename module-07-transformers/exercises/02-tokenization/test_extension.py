"""Tests for Exercise 02 - Text Generation (Part B)."""

import time
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("transformers")

import start


def _mock_generate(text, **kwargs):
    n = kwargs.get("num_return_sequences", 1)
    return [{"generated_text": text + " Generated text."} for _ in range(n)]


@pytest.fixture(autouse=True)
def mock_generators():
    pipes = {}

    def factory(model_name=start.DEFAULT_MODEL):
        if model_name not in pipes:
            pipe = MagicMock(side_effect=_mock_generate)
            pipes[model_name] = pipe
        return pipes[model_name]

    with patch.object(start, "load_generator", side_effect=factory):
        yield pipes


class TestModelComparison:
    def test_distilgpt2_generates(self):
        gen = start.load_generator("distilgpt2")
        result = gen("Test prompt")
        assert len(result) == 1
        assert "generated_text" in result[0]

    def test_gpt2_generates(self):
        gen = start.load_generator("gpt2")
        result = gen("Test prompt")
        assert len(result) == 1
        assert "generated_text" in result[0]
