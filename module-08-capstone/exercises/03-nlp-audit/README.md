# Exercise 03 - NLP Audit Dashboard

Run **every NLP technique from the course** against a corpus and produce a structured audit report. This is the full workup: preprocessing stats, topic discovery, sentiment scan, entity census, and classification probe - all automated in one pipeline.

## Before you start

```bash
pip install -e ".[nlp,local-ml,dev]"
python -m spacy download en_core_web_sm
```

Requires `[nlp]` for spaCy and sklearn. Sentiment scan requires `[local-ml]` (PyTorch + transformers). Skip the sentiment phase if torch is not installed.

Open `start.py`. Each function has a `# TODO` - implement them in order.

## The data

Use any dataset from Exercise 01's `DATASETS.md`, or default to Inkwell statements:

| Key | Source |
| --- | ------ |
| `inkwell` | `data/inkwell/statements.json` (default) |
| `movie_reviews` | `data/public/movie_reviews_sample.json` |
| `sms_spam` | `data/public/sms_spam_sample.json` |
| `custom` | Your own JSON with `{id, text, label}` |

## What you'll build

```bash
python start.py audit --dataset inkwell
```

```
NLP Audit Report - inkwell
==========================

1. Corpus stats
   Documents: 10  |  Avg length: 142 chars  |  Vocabulary: 187 unique tokens

2. Topic discovery (3 topics)
   Topic 0: docks, reeves, pier, warehouse, foggy
   Topic 1: nervous, denied, calm, interview, witness
   Topic 2: case, evidence, ledger, smuggling, customs

3. Sentiment scan
   POSITIVE: 3 (30%)  |  NEGATIVE: 7 (70%)

4. Entity census
   PERSON (8): Margaret Hayes, Reeves, Whitfield, ...
   LOC (5): docks, pier seven, warehouse, ...
   DATE (3): Tuesday, midnight, 1947

5. Classification probe
   Labels: CASE-42, CASE-17, CASE-88
   Baseline F1: 0.83  |  Zero-shot F1: 0.67

Report saved to: audit_inkwell.json
```

## Milestones

| # | Phase | Functions | Modules used |
| - | ----- | --------- | ------------ |
| 1 | Load | `load_corpus()` | - |
| 2 | Stats | `corpus_stats()` | M1, M2 |
| 3 | Topics | `discover_topics()` | M4 |
| 4 | Sentiment | `sentiment_scan()` | M7 |
| 5 | Entities | `entity_census()` | M6 |
| 6 | Classify | `classification_probe()` | M3, M7 |
| 7 | Report | `build_report()`, `print_report()` | - |
| 8 | Serve | `create_audit_app()` (stretch) | M8 |

## Functions to implement

1. **`load_corpus(dataset, path)`** - Load records as `{id, text, label}` dicts. Inkwell uses `case_id` as label.
2. **`corpus_stats(records)`** - Return `{count, avg_length, min_length, max_length, vocab_size, label_balance}`.
3. **`discover_topics(texts, n_topics)`** - Run NMF on TF-IDF. Return list of `{topic_id, top_terms}`.
4. **`sentiment_scan(texts)`** - Run HF `sentiment-analysis` pipeline. Return `{distribution, scores}`.
5. **`entity_census(texts)`** - Run spaCy NER. Return `{label: {entities: [...], count: N}}`.
6. **`classification_probe(records)`** - If records have labels, train TF-IDF baseline and report F1. Return metrics dict.
7. **`build_report(dataset, stats, topics, sentiment, entities, classification)`** - Assemble a complete JSON-serialisable report.
8. **`print_report(report)`** - Formatted CLI output.
9. **`save_report(report, path)`** - Write report to JSON file.

---

## Stretch goal - audit API

```bash
python start.py serve --port 8000
```

Expose `POST /audit` accepting `{dataset, texts}` and returning the full report as JSON. Use FastAPI as in Exercise 01.

---

## Tests

```bash
cd module-08-capstone/exercises/03-nlp-audit
pytest test_start.py -v
```

Tests use the Inkwell statements and mock the HF sentiment pipeline.

## Checklist

- [ ] `load_corpus("inkwell")` returns 10 records with id, text, label
- [ ] `corpus_stats` reports count, lengths, vocabulary size
- [ ] `discover_topics` returns topic terms from NMF
- [ ] `sentiment_scan` returns distribution counts
- [ ] `entity_census` finds PERSON, LOC, DATE entities
- [ ] `classification_probe` returns F1 when labels exist
- [ ] `build_report` produces a JSON-serialisable dict
- [ ] `print_report` prints a readable summary
- [ ] All tests passed
