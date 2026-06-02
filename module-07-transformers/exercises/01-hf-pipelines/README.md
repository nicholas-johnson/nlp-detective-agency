# Exercise 01 - Transformer Inference Lab

Run a **local transformer model** across four NLP tasks on Inkwell witness statements. See model loading, inference timing, and confidence scores in action.

## Before you start

```bash
pip install -e ".[nlp,local-ml,dev]"
```

First run downloads models (~250 MB, cached by Hugging Face). Open `start.py` - each `# TODO` in order.

## The data

| File                                   | Purpose                    |
| -------------------------------------- | -------------------------- |
| `data/inkwell/witness_sentiment.json`  | 14 calm/hostile statements |
| `data/inkwell/statements.json`         | 10 raw witness statements  |
| `data/inkwell/statement_entities.json` | Gold NER for comparison    |

## What you'll build

```bash
python start.py
```

```
Inkwell Investigations - Transformer Inference Lab
==================================================

Loading sentiment-analysis (distilbert-base-uncased-finetuned-sst-2-english)...
Loading ner (dbmdz/bert-large-cased-finetuned-conll03-english)...
Loading zero-shot-classification (facebook/bart-large-mnli)...
Loading summarization (sshleifer/distilbart-cnn-12-6)...

--- Sentiment Analysis (14 statements) ---
  SNT-001 Margaret Hayes: NEGATIVE (0.93) ← hostile
  SNT-002 Thomas Whitfield: NEGATIVE (0.78) ← calm
  ...
  Time: 1.2s

--- Named Entity Recognition (10 statements) ---
  STM-001: PER: Reeves | LOC: docks | DATE: Tuesday
  STM-003: PER: Margaret Hayes, Reeves, Whitfield | LOC: warehouse, Anchor
  ...
  Evidence board (CASE-42): PERSON: {Reeves, Whitfield, ...} LOC: {docks, ...}
  Time: 2.4s

--- Zero-shot Classification ---
  SNT-001: hostile (0.82) > calm (0.18)  actual=hostile ✓
  SNT-004: calm (0.91) > hostile (0.09)  actual=calm ✓
  ...
  Time: 3.1s

--- Summarisation (longest statement) ---
  STM-009 (193 words → 28 words):
  "Two figures were seen near pier seven arguing..."
  Time: 1.8s

Total inference time: 8.5s
```

## Functions

1. `load_model(task)` - load and cache a HF pipeline by task name
2. `analyse_sentiment(records)` - sentiment on witness_sentiment records; return list of `{id, witness, predicted, score, actual}`
3. `extract_entities(statements)` - NER with grouped entities; return per-statement entity dicts
4. `build_evidence_board(entity_results, case_id)` - merge entities across a case
5. `classify_zero_shot(records, labels)` - zero-shot classification; return `{id, predicted, scores, actual}`
6. `summarise_longest(statements)` - summarise the longest statement
7. `run_all()` - orchestrate all tasks with timing

## Part B - Real-world (optional)

```bash
python start.py --real-world
```

SMS spam sentiment + CoNLL NER on bundled public samples.

## Tests

```bash
cd module-07-transformers/exercises/01-hf-pipelines
pytest test_start.py test_extension.py -v
```

## Checklist

- [ ] All four pipelines load and produce results
- [ ] Evidence board merges entities per case
- [ ] Timing is printed for each task
- [ ] All tests pass
