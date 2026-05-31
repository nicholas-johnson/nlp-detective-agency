# Exercise 01 - Archive Themes

Run LDA on the cold-case archive and map each file to its dominant theme. The demo lets you pick one case interactively - this exercise produces a **full batch report** for every file.

## Before you start

```bash
pip install -e ".[nlp,dev]"
```

Open `start.py`. Each function has a `# TODO` - implement them in order.

## The data

Cold-case summaries live in `data/inkwell/cold_cases.json`. Each record has:

| Field     | Description                  |
| --------- | ---------------------------- |
| `id`      | Archive ID (e.g. `ARC-001`)  |
| `case_id` | Linked case (e.g. `CASE-42`) |
| `title`   | Short case title             |
| `summary` | 2–4 sentence summary text    |

There are **27 summaries** across waterfront, financial, surveillance, and neighbourhood themes - but **no labels**. LDA must discover structure unsupervised.

## What you'll build

```bash
python start.py
```

prints topic top-words and per-case assignments:

```
Inkwell Investigations - Archive Themes (LDA)
============================================

Topic 0: dock, warehouse, pier, fog, reeves, quay
...

Case assignments:
  ARC-001 (CASE-42) - topic 0 (0.412)
  ...
```

The `main()` function is already written.

## Functions to implement

### 1. `load_archive(path)`

```python
def load_archive(path: Path) -> list[dict]:
    return json.loads(path.read_text())
```

### 2. `build_dtm(texts)`

LDA works on **word counts**:

```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words="english", min_df=2)
dtm = vectorizer.fit_transform(texts)
return dtm, vectorizer.get_feature_names_out()
```

### 3. `fit_lda(dtm, n_topics=4)`

```python
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=20,
)
lda.fit(dtm)
return lda
```

### 4. `top_words(model, feature_names, n=8)`

For each topic row in `model.components_`, return the top `n` terms by weight:

```python
ranked = row.argsort()[::-1][:n]
words = [(feature_names[i], round(float(row[i]), 4)) for i in ranked]
```

Return `[{"topic_id": 0, "words": [...]}, ...]`.

### 5. `dominant_topics(model, dtm)`

```python
doc_topics = model.transform(dtm)
# per row: argmax index and max weight
```

### 6. `archive_report(records, n_topics=4)`

Combine everything into:

```python
{
    "topics": [...],
    "cases": [
        {"id": "ARC-001", "case_id": "CASE-42", "title": "...", "dominant_topic": 0, "weight": 0.41},
        ...
    ],
}
```

---

## Run it

```bash
python module-04-topic-modelling/exercises/01-archive-themes/start.py
```

## Run the tests

```bash
cd module-04-topic-modelling/exercises/01-archive-themes
pytest test_start.py -v
```

---

## Optional hints

<details>
<summary>Hint: why CountVectorizer for LDA?</summary>

LDA models document generation from word **counts**. TF-IDF can work but count data is the standard input. NMF (Exercise 02) uses TF-IDF instead.

</details>

<details>
<summary>Hint: min_df=2</summary>

Terms appearing in only one document are usually noise for topic discovery. `min_df=2` requires a word in at least two summaries before it enters the vocabulary.

</details>

<details>
<summary>Hint: components_ shape</summary>

`lda.components_` has shape `(n_topics, vocab_size)`. Row `i` is the word distribution for topic `i`. Higher values mean the word is more characteristic of that topic.

</details>

<details>
<summary>Hint: soft assignments</summary>

`transform(dtm)` returns weights that sum to ~1 per document. A case can have weight on multiple topics - `dominant_topic` is just the argmax.

</details>

---

## Checklist

- [ ] `load_archive` reads the JSON archive
- [ ] `build_dtm` uses `CountVectorizer(stop_words="english", min_df=2)`
- [ ] `fit_lda` uses `random_state=42`
- [ ] `top_words` returns 8 words per topic
- [ ] `dominant_topics` returns one assignment per document
- [ ] `archive_report` returns topics and cases lists
- [ ] `python start.py` prints topics and assignments
- [ ] `pytest test_start.py -v` - all passed
