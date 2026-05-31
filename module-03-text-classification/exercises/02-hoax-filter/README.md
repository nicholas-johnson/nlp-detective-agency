# Exercise 02 - Hoax Filter

The tip line is flooded with hoaxes. Credible tips mention specifics - times, places, names. Hoaxes are vague, sensational, or contradictory.

The demo runs an interactive classifier shootout. This exercise **automates** the comparison, trains the best model, and lists **false negatives** - hoaxes that slip through as credible.

## Before you start

```bash
pip install -e ".[nlp,dev]"
```

Open `start.py`. Each function has a `# TODO` - implement them in order.

## The data

Anonymous tip-offs live in `data/inkwell/tips.json`. Each record has:

| Field   | Description             |
| ------- | ----------------------- |
| `id`    | Tip ID (e.g. `TIP-001`) |
| `text`  | Tip letter text         |
| `label` | `credible` or `hoax`    |

There are **18 tips** - 10 credible, 8 hoax - with slight class imbalance to motivate precision/recall discussion.

## What you'll build

```bash
python start.py
```

prints a classifier shootout table, confusion matrix, and slipped-through hoax IDs:

```
Inkwell Investigations - Hoax Filter Report
============================================

Classifier            F1 mean   F1 std
----------------------------------------
Logistic Regression      0.850    0.120
Linear SVM               0.820    0.140
Naive Bayes              0.780    0.160

Best classifier: Logistic Regression

Confusion matrix (rows=actual, cols=predicted):
              credible  hoax
  credible           3     0
  hoax               1     1

Hoaxes that slipped through (1): TIP-007
```

The `main()` function is already written.

## Functions to implement

### 1. `load_tips(path)`

```python
def load_tips(path: Path) -> list[dict]:
    return json.loads(path.read_text())
```

### 2. `build_pipeline(classifier_name)`

Map short names to classifiers inside a Pipeline:

```python
classifiers = {
    "nb": MultinomialNB(),
    "lr": LogisticRegression(max_iter=1000, random_state=42),
    "svm": LinearSVC(dual="auto", random_state=42),
}
if classifier_name not in classifiers:
    raise ValueError(f"Unknown classifier: {classifier_name}")

return Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", classifiers[classifier_name]),
])
```

### 3. `compare_classifiers(records)`

Use **5-fold cross-validation** with F1 scoring:

```python
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score

texts = [r["text"] for r in records]
labels = [r["label"] for r in records]
hoax_f1 = make_scorer(f1_score, pos_label="hoax")

for key in ("nb", "lr", "svm"):
    pipeline = build_pipeline(key)
    scores = cross_val_score(pipeline, texts, labels, cv=5, scoring=hoax_f1)
    # append {"name": "...", "key": key, "f1_mean": scores.mean(), "f1_std": scores.std()}
```

Return results **sorted by `f1_mean` descending**.

### 4. `confusion(labels_true, labels_pred)`

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(labels_true, labels_pred, labels=["credible", "hoax"])
return cm.tolist()
```

Row order: credible first, hoax second. A false negative (hoax predicted credible) appears in `cm[1][0]`.

### 5. `false_negatives(records, pipeline, X_test, y_test)`

Predict on test texts. Return **sorted** list of tip IDs where actual is `hoax` but predicted is `credible`:

```python
preds = pipeline.predict(X_test)
id_by_text = {r["text"]: r["id"] for r in records}
slipped = []
for text, actual, pred in zip(X_test, y_test, preds):
    if actual == "hoax" and pred == "credible":
        slipped.append(id_by_text[text])
return sorted(slipped)
```

### 6. `hoax_report(records)`

Full workflow:

1. `compare_classifiers` → pick best by F1
2. Stratified train/test split (`random_state=42`, `test_size=0.25`)
3. Train best pipeline on train
4. Predict on test
5. Return:

```python
{
    "comparison": [...],
    "best_classifier": "Logistic Regression",
    "confusion_matrix": [[...], [...]],
    "slipped_through": ["TIP-007", ...],
}
```

---

## Run it

```bash
python module-03-text-classification/exercises/02-hoax-filter/start.py
```

## Run the tests

```bash
cd module-03-text-classification/exercises/02-hoax-filter
pytest test_start.py -v
```

---

## Optional hints

<details>
<summary>Hint: cross_val_score and F1</summary>

With string labels, `scoring="f1"` fails - use `make_scorer(f1_score, pos_label="hoax")` so cross-validation measures hoax recall/precision. All three classifiers are compared on the same metric.

</details>

<details>
<summary>Hint: LinearSVC and dual="auto"</summary>

For wide sparse TF-IDF matrices (many features, few documents), `dual="auto"` lets sklearn pick the efficient formulation. Set `random_state=42` for reproducibility where supported.

</details>

<details>
<summary>Hint: false negatives vs false positives</summary>

For a **hoax filter**, false negatives are dangerous - a hoax marked credible wastes detective time. False positives (credible marked hoax) discard good leads. Tune thresholds based on which error costs more.

</details>

<details>
<summary>Hint: confusion matrix layout</summary>

```python
confusion_matrix(y_true, y_pred, labels=["credible", "hoax"])
```

```
                 predicted
              credible  hoax
actual credible    TN       FP
       hoax        FN       TP   ← FN = slipped through
```

</details>

<details>
<summary>Hint: LogisticRegression max_iter</summary>

TF-IDF features can make convergence slow. `max_iter=1000` avoids warnings on small datasets. If you see convergence warnings, increase further.

</details>

---

## Checklist

- [ ] `load_tips` reads the JSON archive
- [ ] `build_pipeline` supports `nb`, `lr`, `svm` and raises on unknown names
- [ ] `compare_classifiers` returns 3 results sorted by F1 mean
- [ ] `confusion` returns a 2×2 nested list
- [ ] `false_negatives` returns only hoax tips predicted as credible
- [ ] `hoax_report` combines comparison, best model, matrix, and slipped IDs
- [ ] `python start.py` prints shootout table and slipped hoaxes
- [ ] `pytest test_start.py -v` - all passed
