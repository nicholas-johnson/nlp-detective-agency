# Exercise 01 - HF Pipelines

Run Hugging Face **pipelines** for sentiment, NER, and zero-shot classification on Inkwell data. Compare transformer NER to Module 6 gold labels.

## Before you start

```bash
pip install -e ".[nlp,local-ml,dev]"
```

First pipeline run downloads models (~250 MB). Open `start.py` - each `# TODO` in order.

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

## Functions

1. `load_pipeline(task)` - cache HF pipeline by task name
2. `classify_sentiment(texts)` - sentiment-analysis pipeline
3. `extract_entities_hf(texts)` - NER with `grouped_entities=True`
4. `zero_shot_classify(texts, labels)` - zero-shot-classification
5. `compare_ner_to_gold(hf_entities, gold_entities)` - recall metric

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
