"""
Capstone Exercise 02 - Inkwell Case Chatbot
Retrieve witness statements and answer questions with a HF chat model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "inkwell"
STATEMENTS_PATH = DATA_DIR / "statements.json"
COLD_CASES_PATH = DATA_DIR / "cold_cases.json"
ENTITIES_PATH = DATA_DIR / "statement_entities.json"

DEFAULT_MODEL = "microsoft/DialoGPT-medium"

_chat_model = None
_nlp = None


def load_case_documents(case_id: str) -> list[dict]:
    """Load statements and cold case records for a case.

    Return list of {id, source, text} dicts.
    source is 'statement' or 'archive'.
    """
    # TODO
    raise NotImplementedError


def build_tfidf_index(documents: list[dict]):
    """Fit TfidfVectorizer on document texts.

    Return (vectorizer, tfidf_matrix).
    """
    # TODO
    raise NotImplementedError


def retrieve_context(
    query: str,
    vectorizer,
    tfidf_matrix,
    documents: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Find the top_k most relevant documents by cosine similarity.

    Return documents sorted by descending similarity.
    """
    # TODO
    raise NotImplementedError


def build_prompt(
    question: str,
    context_docs: list[dict],
    case_id: str,
) -> str:
    """Assemble a prompt: system instruction + context documents + question."""
    # TODO
    raise NotImplementedError


def load_chat_model(model_name: str = DEFAULT_MODEL):
    """Load a HF text-generation pipeline. Cache in module-level _chat_model."""
    # TODO
    raise NotImplementedError


def generate_answer(model, prompt: str, max_new_tokens: int = 200) -> str:
    """Generate a response from the chat model. Return generated text only."""
    # TODO
    raise NotImplementedError


def extract_response_entities(text: str) -> dict[str, list[str]]:
    """Run spaCy NER on text. Return {label: [entity_texts]}."""
    # TODO
    raise NotImplementedError


def validate_entities(
    response_entities: dict[str, list[str]],
    case_documents: list[dict],
) -> dict[str, list[str]]:
    """Check which response entities appear in case data.

    Return {grounded: [...], ungrounded: [...]}.
    """
    # TODO
    raise NotImplementedError


def format_answer(answer: str, entities: dict[str, list[str]]) -> str:
    """Format the answer with entity tags for display."""
    # TODO
    raise NotImplementedError


def chat_loop(case_id: str, model_name: str = DEFAULT_MODEL) -> None:
    """Interactive CLI chat loop."""
    # TODO
    raise NotImplementedError


def run_index(case_id: str) -> None:
    """CLI: show indexed documents for a case."""
    # TODO
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Inkwell Case Chatbot")
    parser.add_argument(
        "command",
        choices=["index", "chat"],
        nargs="?",
        default="chat",
    )
    parser.add_argument("--case", default="CASE-42")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.command == "index":
        run_index(args.case)
    elif args.command == "chat":
        chat_loop(args.case, args.model)


if __name__ == "__main__":
    main()
