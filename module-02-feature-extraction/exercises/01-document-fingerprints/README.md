# Exercise 01 — Document Fingerprints

The lab needs a count-based fingerprint card for every witness statement in a case. Raw word counts reveal what each witness talked about most — the first step in comparing statements.

This exercise uses **CountVectorizer** (Bag-of-Words). TF-IDF and similarity come in Exercise 02.

## Before you start

```bash
pip install -e ".[nlp,dev]"
```

Open `start.py`. Each function has a `# TODO` — implement them in order.

## The data

Witness statements live in `data/inkwell/statements.json`. Each record has `id`, `case_id`, `witness`, and `raw_text`.

`CASE-42` has **4 statements** — a good case to test with.

## What you'll build

```bash
python start.py CASE-42
```

prints fingerprint cards like:

```
Inkwell Investigations — Fingerprints for CASE-42
============================================================

STM-001 — Margaret Hayes
  saw(2), docks(1), tuesday(1), ...

STM-002 — Thomas Whitfield
  dock(1), empty(1), night(1), ...
```

The `main()` function is already written.

## Functions to implement

### 1. `load_statements(path)`

```python
def load_statements(path: Path) -> list[dict]:
    return json.loads(path.read_text())
```

### 2. `texts_for_case(statements, case_id)`

Filter the archive and return two parallel lists:

```python
matched = [s for s in statements if s["case_id"] == case_id]
doc_ids = [s["id"] for s in matched]
texts = [s["raw_text"] for s in matched]
return doc_ids, texts
```

Return `([], [])` when no statements match.

### 3. `build_bow_matrix(texts)`

Use scikit-learn's `CountVectorizer`:

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(texts)
feature_names = vectorizer.get_feature_names_out()
return matrix, feature_names
```

- `stop_words="english"` removes common words like "the", "was", "and"
- `matrix` is a **sparse** scipy matrix — shape `(num_documents, vocab_size)`
- `feature_names` is an array of vocabulary terms

### 4. `top_terms(matrix, feature_names, doc_index, n=5)`

Extract the row for one document and return the top `n` terms by count:

```python
row = matrix[doc_index].toarray().flatten()
ranked = row.argsort()[::-1]   # indices highest to lowest
# return (feature_names[i], int(row[i])) for non-zero entries, take first n
```

### 5. `fingerprint_report(statements, case_id)`

Combine everything:

1. Get `doc_ids, texts` for the case
2. If empty, return `[]`
3. Build the BoW matrix
4. For each statement, call `top_terms` and build a card dict
5. Return cards sorted by `id`

```python
{
    "id": "STM-001",
    "witness": "Margaret Hayes",
    "top_terms": [("saw", 2), ("docks", 1), ...],
}
```

---

## Run it

```bash
python module-02-feature-extraction/exercises/01-document-fingerprints/start.py CASE-42
```

## Run the tests

```bash
cd module-02-feature-extraction/exercises/01-document-fingerprints
pytest test_start.py -v
```

---

## Optional hints

<details>
<summary>Hint: sparse matrices</summary>

`matrix.toarray()` converts to a dense numpy array — fine for small exercises. For large corpora, use sparse indexing:

```python
row = matrix.getrow(doc_index)
coo = row.tocoo()
```

Never call `.toarray()` on a matrix with millions of features.

</details>

<details>
<summary>Hint: fit vs transform</summary>

`fit_transform` learns the vocabulary **and** converts in one step. Use this when building from scratch. In production, `fit` on training data and `transform` on test data separately.

</details>

<details>
<summary>Hint: sorting by count</summary>

`row.argsort()[::-1]` returns indices from highest to lowest value. Skip zero-count terms — they are not interesting fingerprint material.

</details>

<details>
<summary>Hint: STM-002 should mention the dock</summary>

Thomas Whitfield's statement is *"The docks were EMPTY that night..."* After vectorisation, `dock` or `docks` should appear in his top terms. The test checks for this.

</details>

---

## Checklist

- [ ] `load_statements` reads the JSON archive
- [ ] `texts_for_case` returns parallel id and text lists
- [ ] `build_bow_matrix` uses `CountVectorizer(stop_words="english")`
- [ ] `top_terms` returns `(term, count)` pairs sorted by count
- [ ] `fingerprint_report` returns cards sorted by `id`
- [ ] `python start.py CASE-42` prints fingerprint cards
- [ ] `pytest test_start.py -v` — all passed
