"""
Capstone Exercise 02 - Inkwell Case Chatbot (solution)
Retrieve witness statements and answer questions with a HF chat model.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "inkwell"
STATEMENTS_PATH = DATA_DIR / "statements.json"
COLD_CASES_PATH = DATA_DIR / "cold_cases.json"
ENTITIES_PATH = DATA_DIR / "statement_entities.json"

DEFAULT_MODEL = "microsoft/DialoGPT-medium"

_chat_model = None
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def load_case_documents(case_id: str) -> list[dict]:
    """Load statements and cold case records for a case."""
    documents = []

    statements = json.loads(STATEMENTS_PATH.read_text())
    for s in statements:
        if s["case_id"] == case_id:
            documents.append({
                "id": s["id"],
                "source": "statement",
                "text": s["raw_text"],
            })

    cold_cases = json.loads(COLD_CASES_PATH.read_text())
    for c in cold_cases:
        if c["case_id"] == case_id:
            documents.append({
                "id": c["id"],
                "source": "archive",
                "text": f"{c['title']}. {c['summary']}",
            })

    return documents


def build_tfidf_index(documents: list[dict]):
    """Fit TfidfVectorizer on document texts."""
    texts = [d["text"] for d in documents]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


def retrieve_context(
    query: str,
    vectorizer,
    tfidf_matrix,
    documents: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """Find the top_k most relevant documents by cosine similarity."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked = sorted(
        zip(scores, documents, strict=True),
        key=lambda x: x[0],
        reverse=True,
    )
    return [doc for _, doc in ranked[:top_k]]


def build_prompt(
    question: str,
    context_docs: list[dict],
    case_id: str,
) -> str:
    """Assemble a prompt with system instruction, context, and question."""
    context_block = "\n\n".join(
        f"[{doc['id']} ({doc['source']})] {doc['text']}"
        for doc in context_docs
    )
    return (
        f"You are an investigative assistant analysing {case_id}. "
        f"Answer the question using ONLY the case documents below. "
        f"If the documents do not contain the answer, say so.\n\n"
        f"--- Case documents ---\n{context_block}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def load_chat_model(model_name: str = DEFAULT_MODEL):
    """Load a HF text-generation pipeline."""
    global _chat_model
    if _chat_model is None:
        from transformers import pipeline
        _chat_model = pipeline(
            "text-generation",
            model=model_name,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
        )
    return _chat_model


def generate_answer(model, prompt: str, max_new_tokens: int = 200) -> str:
    """Generate a response and strip the prompt prefix."""
    result = model(prompt, max_new_tokens=max_new_tokens, return_full_text=False)
    return result[0]["generated_text"].strip()


def extract_response_entities(text: str) -> dict[str, list[str]]:
    """Run spaCy NER on the response text."""
    nlp = _get_nlp()
    doc = nlp(text)
    entities: dict[str, list[str]] = defaultdict(list)
    for ent in doc.ents:
        label = ent.label_
        if label in ("GPE", "FAC"):
            label = "LOC"
        if ent.text not in entities[label]:
            entities[label].append(ent.text)
    return dict(entities)


def validate_entities(
    response_entities: dict[str, list[str]],
    case_documents: list[dict],
) -> dict[str, list[str]]:
    """Check which response entities appear in case documents."""
    corpus = " ".join(d["text"] for d in case_documents).lower()
    grounded = []
    ungrounded = []
    for label, names in response_entities.items():
        for name in names:
            if name.lower() in corpus:
                grounded.append(name)
            else:
                ungrounded.append(name)
    return {"grounded": grounded, "ungrounded": ungrounded}


def format_answer(answer: str, entities: dict[str, list[str]]) -> str:
    """Format the answer with entity tags for display."""
    entity_tags = []
    for label, names in sorted(entities.items()):
        entity_tags.append(f"[{label}: {', '.join(names)}]")
    suffix = " ".join(entity_tags)
    if suffix:
        return f"{answer}\n{suffix}"
    return answer


def chat_loop(case_id: str, model_name: str = DEFAULT_MODEL) -> None:
    """Interactive CLI chat loop."""
    documents = load_case_documents(case_id)
    if not documents:
        print(f"No documents found for {case_id}.")
        return

    vectorizer, tfidf_matrix = build_tfidf_index(documents)
    model = load_chat_model(model_name)

    n_statements = sum(1 for d in documents if d["source"] == "statement")
    n_archive = sum(1 for d in documents if d["source"] == "archive")

    print(f"\nInkwell Case Chatbot - {case_id}")
    print("=" * 40)
    print(f"{len(documents)} documents loaded ({n_statements} statements, {n_archive} archive records).")
    print("Type 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if not question or question.lower() in ("quit", "exit", "q"):
            print("Session ended.")
            break

        context_docs = retrieve_context(question, vectorizer, tfidf_matrix, documents)
        prompt = build_prompt(question, context_docs, case_id)
        answer = generate_answer(model, prompt)
        entities = extract_response_entities(answer)
        validation = validate_entities(entities, documents)

        print(f"Bot: {format_answer(answer, entities)}")
        if validation["ungrounded"]:
            print(f"  (ungrounded names: {', '.join(validation['ungrounded'])})")
        print()


def run_index(case_id: str) -> None:
    """CLI: show indexed documents for a case."""
    documents = load_case_documents(case_id)
    print(f"\nInkwell Case Index - {case_id}")
    print("=" * 40)
    print(f"{len(documents)} documents:\n")
    for doc in documents:
        print(f"  [{doc['id']}] ({doc['source']}) {doc['text'][:80]}...")
    print()


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
