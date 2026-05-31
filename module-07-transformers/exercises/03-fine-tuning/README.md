# Exercise 03 - Fine-Tuning (optional)

Fine-tune **DistilBERT** on Inkwell witness sentiment (calm/hostile). Compare to Module 3's TF-IDF + Naive Bayes baseline.

## Before you start

```bash
pip install -e ".[nlp,local-ml,dev]"
```

Requires **PyTorch** and several minutes on CPU. Skip if disk/GPU time is limited.

Open `start.py`. Each `# TODO` in order.

## The data

| File                                    | Purpose                         |
| --------------------------------------- | ------------------------------- |
| `data/inkwell/witness_sentiment.json`   | 14 labelled statements (Part A) |
| `data/public/movie_reviews_sample.json` | 40 reviews (Part B)             |

## What you'll build

```bash
python start.py
```

## Functions

1. `load_sentiment_data(path)` / `split_data(records)`
2. `sklearn_baseline(texts, labels)` - TF-IDF + NB from Module 3
3. `tokenize_dataset(tokenizer, texts, labels)` - HF Dataset
4. `build_trainer(model, tokenizer, train_ds, eval_ds)`
5. `train_and_evaluate(records)` - train 3 epochs, return eval metrics
6. `compare_to_baseline(hf_metrics, baseline_metrics)`

Training args: 3 epochs, batch size 4, lr 5e-5, max_length 128.

## Part B - Real-world (optional)

```bash
python start.py --real-world
```

Fine-tune on movie reviews (pos → calm, neg → hostile mapping).

## Tests

```bash
cd module-07-transformers/exercises/03-fine-tuning
pytest test_start.py test_extension.py -v
```

Tests mock training - no GPU required for pytest.
