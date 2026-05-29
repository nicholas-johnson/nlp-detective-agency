# Exercise 01 — Sentiment Triage

Before a detective interviews a witness, the system flags whether the statement reads **calm** or **hostile**. That helps prioritise follow-up and set interview tone.

The demo lets you pick one statement and predict interactively. This exercise trains a classifier, evaluates on a **held-out test set**, and returns structured metrics.

## Before you start

```bash
pip install -e ".[nlp,dev]"
```

Open `start.py`. Each function has a `# TODO` — implement them in order.

## The data

Labeled witness statements live in `data/inkwell/witness_sentiment.json`. Each record has:

| Field       | Description                          |
| ----------- | ------------------------------------ |
| `id`        | Statement ID (e.g. `WS-001`)         |
| `witness`   | Witness name                         |
| `text`      | Statement text                       |
| `sentiment` | `calm` or `hostile`                  |

There are **14 records** — 7 calm, 7 hostile — drawn from Inkwell case files and a few synthetic statements.

## What you'll build

```bash
python start.py
```

prints triage metrics and any misclassifications:

```
Inkwell Investigations — Sentiment Triage
============================================
Accuracy: 0.750
F1 (hostile): 0.667

Misclassified (1):
  WS-003 Thomas Whitfield: actual=calm, predicted=hostile
```

The `main()` function is already written.

## Functions to implement

### 1. `load_sentiment_data(path)`

```python
def load_sentiment_data(path: Path) -> list[dict]:
    return json.loads(path.read_text())
```

### 2. `split_data(records)`

Extract texts and labels, then split with stratification:

```python
texts = [r["text"] for r in records]
labels = [r["sentiment"] for r in records]
return train_test_split(
    texts, labels,
    test_size=0.25,
    random_state=42,
    stratify=labels,
)
```

**Stratify** keeps the same calm/hostile ratio in train and test. With 14 records, you get 11 train / 3 test (or 10/4 depending on rounding — sklearn handles this).

### 3. `build_sentiment_pipeline()`

Combine vectorisation and classification in one object:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

return Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB()),
])
```

Pipelines ensure TF-IDF is fit on training data only — no leakage.

### 4. `train_and_evaluate(records)`

Full workflow:

1. Split records (use index-based split if you need to attach `id` and `witness` to predictions)
2. Fit the pipeline on train
3. Predict on test
4. Return a dict:

```python
{
    "accuracy": accuracy_score(y_test, preds),
    "f1": f1_score(y_test, preds, pos_label="hostile"),
    "predictions": [
        {"id": "...", "witness": "...", "actual": "...", "predicted": "..."},
        ...
    ],
}
```

Use `pos_label="hostile"` for F1 — we care about catching hostile statements.

### 5. `triage_report(records)`

Delegate to `train_and_evaluate` — it is the public entry point for the full report.

---

## Run it

```bash
python module-03-text-classification/exercises/01-sentiment-triage/start.py
```

## Run the tests

```bash
cd module-03-text-classification/exercises/01-sentiment-triage
pytest test_start.py -v
```

---

## Optional hints

<details>
<summary>Hint: attaching metadata to predictions</summary>

`train_test_split` on texts and labels loses record IDs. Split **indices** instead:

```python
indices = list(range(len(records)))
train_idx, test_idx = train_test_split(
    indices, test_size=0.25, random_state=42, stratify=labels,
)
X_train = [texts[i] for i in train_idx]
# ...
predictions = [
    {"id": records[i]["id"], "witness": records[i]["witness"], ...}
    for i, pred in zip(test_idx, preds)
]
```

</details>

<details>
<summary>Hint: stratify requires at least 2 per class</summary>

Stratified splitting needs at least one sample per class in each split. Our balanced dataset (7/7) is fine. With tiny or imbalanced data, stratify can fail — use `stratify=None` only when you must.

</details>

<details>
<summary>Hint: F1 for binary classification</summary>

For two classes, `f1_score(y_true, y_pred, pos_label="hostile")` measures precision/recall for the hostile class. Without `pos_label`, sklearn averages both classes.

</details>

<details>
<summary>Hint: why MultinomialNB?</summary>

Naive Bayes works well with sparse TF-IDF features and trains fast on small datasets. It is a solid baseline before trying Logistic Regression or SVM in Exercise 02.

</details>

---

## Checklist

- [ ] `load_sentiment_data` reads the JSON archive
- [ ] `split_data` uses `stratify` and `random_state=42`
- [ ] `build_sentiment_pipeline` has `tfidf` and `clf` steps
- [ ] `train_and_evaluate` returns accuracy, F1, and per-test predictions
- [ ] `triage_report` delegates to `train_and_evaluate`
- [ ] `python start.py` prints metrics and misclassifications
- [ ] `pytest test_start.py -v` — all passed
