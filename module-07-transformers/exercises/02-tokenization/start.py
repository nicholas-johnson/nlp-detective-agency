"""
Exercise 02 - Text Generation
Load a local causal LM and generate continuations of witness statements.
"""

import argparse
import json
import time
from pathlib import Path

STATEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
)

DEFAULT_MODEL = "distilgpt2"

_generators: dict = {}


def load_generator(model_name: str = DEFAULT_MODEL):
    """Load and cache a text-generation pipeline for the given model."""
    # TODO: use transformers.pipeline("text-generation", model=model_name)
    raise NotImplementedError


def load_json(path: Path) -> list[dict]:
    """Load a JSON array from disk."""
    # TODO
    raise NotImplementedError


def continue_statement(text: str, max_new_tokens: int = 50) -> str:
    """Generate a single continuation of the given text.

    Return only the generated portion (not the original prompt).
    """
    # TODO: call load_generator, generate, strip prompt from output
    raise NotImplementedError


def generate_variants(text: str, n: int = 3, temperature: float = 0.7) -> list[str]:
    """Generate n different continuations at the given temperature.

    Return list of generated texts (without prompt).
    """
    # TODO: use do_sample=True, temperature=temperature, num_return_sequences=n
    raise NotImplementedError


def interrogation_prompt(witness_name: str, context: str, max_new_tokens: int = 40) -> str:
    """Build an interrogation prompt and generate a follow-up question.

    Prompt format:
      "Detective's follow-up question for {witness_name}, who said: '{context}'\nQuestion:"

    Return the generated question text.
    """
    # TODO
    raise NotImplementedError


def batch_generate(statements: list[dict], max_new_tokens: int = 40) -> list[dict]:
    """Generate continuations for all statements.

    Return list of {id, witness, prompt_snippet, continuation}.
    """
    # TODO: use first 80 chars of raw_text as prompt
    raise NotImplementedError


def run_inkwell() -> None:
    """Run generation tasks on Inkwell data with timing."""
    # TODO: load statements, run continue_statement, generate_variants,
    # interrogation_prompt, batch_generate. Print formatted output.
    raise NotImplementedError


def run_real_world() -> None:
    """Compare distilgpt2 vs gpt2 generation speed and output."""
    # TODO: time both models on same prompts, print comparison
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
