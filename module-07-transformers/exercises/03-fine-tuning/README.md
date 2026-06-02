# Exercise 03 - Fine-Tuning

Fine-tune **DistilBERT** on Inkwell witness sentiment (calm/hostile). Watch the model learn, compare before/after predictions, and beat the sklearn baseline.

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

```
Inkwell Investigations - Fine-Tuning Lab
========================================

--- Before training (pre-trained DistilBERT) ---
  SNT-001 Margaret Hayes: NEGATIVE (0.93)  actual=hostile
  SNT-002 Thomas Whitfield: NEGATIVE (0.78)  actual=calm
  SNT-004 Harold Finch: POSITIVE (0.61)  actual=calm

--- Sklearn baseline (TF-IDF + Naive Bayes) ---
  Accuracy: 0.750 | F1 (hostile): 0.667

--- Fine-tuning DistilBERT (3 epochs) ---
  Epoch 1/3: loss=0.72 eval_acc=0.50
  Epoch 2/3: loss=0.45 eval_acc=0.75
  Epoch 3/3: loss=0.21 eval_acc=1.00

--- After training (fine-tuned) ---
  SNT-001 Margaret Hayes: hostile (0.91)  actual=hostile ✓
  SNT-002 Thomas Whitfield: calm (0.84)  actual=calm ✓
  SNT-004 Harold Finch: calm (0.95)  actual=calm ✓

--- Comparison ---
  Sklearn baseline:  accuracy=0.750  F1=0.667
  Fine-tuned BERT:   accuracy=1.000  F1=1.000
```

## Functions

1. `load_sentiment_data(path)` - load JSON records
2. `split_data(records)` - stratified train/test split
3. `sklearn_baseline(texts, labels)` - TF-IDF + NB baseline metrics
4. `predict_samples(model, tokenizer, texts)` - predict sentiment for sample texts
5. `tokenize_dataset(tokenizer, texts, labels)` - build HF Dataset
6. `build_trainer(model, tokenizer, train_ds, eval_ds)` - configure Trainer
7. `train_and_evaluate(records)` - train 3 epochs, return metrics
8. `compare_to_baseline(hf_metrics, baseline_metrics)` - side-by-side results

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

## Checklist

- [ ] `predict_samples` shows predictions before and after training
- [ ] `sklearn_baseline` runs and returns accuracy + F1
- [ ] `train_and_evaluate` completes 3 epochs
- [ ] `compare_to_baseline` prints side-by-side metrics
- [ ] All tests pass
