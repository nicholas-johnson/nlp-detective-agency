# Exercise 02 - Inkwell Case Chatbot

Build a **conversational assistant** that answers questions about Inkwell cases. It retrieves relevant witness statements, feeds them as context to a Hugging Face text-generation model, and responds in natural language.

## Before you start

```bash
pip install -e ".[nlp,local-ml,dev]"
python -m spacy download en_core_web_sm
```

Requires PyTorch (`[local-ml]`). First run downloads a chat model (~500 MB–1.5 GB, cached by Hugging Face).

Open `start.py`. Each function has a `# TODO` - implement them in order.

## The data

| File | Purpose |
| ---- | ------- |
| `data/inkwell/statements.json` | Witness statements (10 across 3 cases) |
| `data/inkwell/cold_cases.json` | Archived case summaries (27 records) |
| `data/inkwell/statement_entities.json` | Gold NER labels for entity validation |

## What you'll build

```bash
python start.py chat --case CASE-42
```

```
Inkwell Case Chatbot - CASE-42
==============================
7 documents loaded (4 statements, 3 archive records).

You: Who was seen near the docks?
Bot: Based on the witness statements, Margaret Hayes (STM-001) reported
     seeing a man near the docks on Tuesday, accompanied by someone
     called Reeves.
     [PERSON: Margaret Hayes, Reeves] [LOC: docks]

You: quit
```

## Milestones

| # | Phase | Functions | CLI |
| - | ----- | --------- | --- |
| 1 | Index | `load_case_documents()` | `python start.py index --case CASE-42` |
| 2 | Retrieve | `retrieve_context()` | (used by chat) |
| 3 | Prompt | `build_prompt()` | (used by chat) |
| 4 | Generate | `load_chat_model()`, `generate_answer()` | (used by chat) |
| 5 | Ground | `extract_response_entities()` | (used by chat) |
| 6 | Chat | `chat_loop()` | `python start.py chat --case CASE-42` |
| 7 | Guardrails | `validate_entities()` | (used by chat) |

## Functions to implement

1. **`load_case_documents(case_id)`** - Load statements and cold case records for a case. Return a list of `{id, source, text}` dicts.
2. **`build_tfidf_index(documents)`** - Fit a `TfidfVectorizer` on document texts. Return `(vectorizer, tfidf_matrix)`.
3. **`retrieve_context(query, vectorizer, tfidf_matrix, documents, top_k)`** - Cosine-similarity retrieval. Return the `top_k` most relevant documents.
4. **`build_prompt(question, context_docs, case_id)`** - Assemble a prompt string: system instruction + retrieved context + user question.
5. **`load_chat_model(model_name)`** - Load a HF `pipeline("text-generation", ...)`. Cache globally.
6. **`generate_answer(model, prompt, max_new_tokens)`** - Generate a response. Return the generated text only (strip the prompt).
7. **`extract_response_entities(text)`** - Run spaCy NER on the response. Return `{label: [texts]}`.
8. **`validate_entities(response_entities, case_documents)`** - Check which entities in the response actually appear in the case data. Return `{grounded: [...], ungrounded: [...]}`.
9. **`chat_loop(case_id)`** - Interactive CLI loop: read question, retrieve, generate, display with entity highlights.

## Model choice

Use any HF text-generation model that fits your hardware:

| Model | Size | Notes |
| ----- | ---- | ----- |
| `microsoft/DialoGPT-medium` | ~350 MB | Lightweight, good for CPU |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~2.2 GB | Better quality, needs GPU or patience |
| `HuggingFaceH4/zephyr-7b-beta` | ~14 GB | High quality, needs GPU |

The default in `start.py` is `microsoft/DialoGPT-medium`. Change via `--model`.

## Stretch goals

- **Conversation memory** - maintain a rolling context window of previous Q&A turns.
- **Entity-aware retrieval** - run NER on the query, boost documents mentioning the same entities.
- **FastAPI endpoint** - expose `POST /ask` that accepts `{case_id, question}` and returns `{answer, entities, sources}`.

## Tests

```bash
cd module-08-capstone/exercises/02-case-chatbot
pytest test_start.py -v
```

Tests mock the chat model so they run without GPU or model downloads.

## Checklist

- [ ] `load_case_documents("CASE-42")` returns statements + cold case records
- [ ] `retrieve_context` returns relevant documents sorted by similarity
- [ ] `build_prompt` assembles a well-structured prompt with context
- [ ] `generate_answer` returns generated text (not the prompt)
- [ ] `extract_response_entities` finds NER spans in the response
- [ ] `validate_entities` flags entity names not found in case data
- [ ] `chat_loop` runs interactively with entity highlights
- [ ] All tests passed
