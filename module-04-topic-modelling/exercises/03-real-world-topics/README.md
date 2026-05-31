# Exercise 03 - Real-World Topics

Exercises 01 and 02 discovered themes in the Inkwell cold-case archive - a synthetic detective corpus. Now apply the same pipeline to a **real public dataset** and measure how well unsupervised topics align with actual editorial categories.

The dataset is a sample of **40 articles** from the [20 Newsgroups](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html) corpus - 10 articles each from four newsgroups:

- `rec.sport.baseball`
- `sci.med`
- `sci.space`
- `talk.politics.misc`

No labels are used during topic discovery. Afterwards, you compare discovered topics against the real categories using **purity** and a **contingency matrix**.

## Before you start

```bash
pip install -e ".[nlp,dev]"
```

Open `start.py`. Each function has a `# TODO` - implement them in order.

## The data

Articles live in `data/public/newsgroups_sample.json`. Each record has:

| Field      | Description                            |
| ---------- | -------------------------------------- |
| `id`       | Article ID (e.g. `NG-001`)             |
| `category` | True newsgroup name (for evaluation)   |
| `text`     | Article body (headers/footers removed) |

The `category` field is **not used during training** - only during evaluation.

## What you'll build

```bash
python start.py
```

prints topic words, per-topic purity, and a contingency matrix:

```
Topic Modelling - 20 Newsgroups Audit
============================================

Topic 0: game, team, players, hit, season, runs
Topic 1: space, nasa, launch, orbit, shuttle, earth
...

Purity per topic:
  Topic 0: rec.sport.baseball - 80% (8/10)
  ...

Average purity: 72%

Contingency matrix (topic x category):
            rec.spor   sci.med  sci.spac  talk.pol
  Topic 0         8         1         0         1
  ...
```

The `main()` function is already written.

## Functions to implement

### 1. `load_articles(path)`

```python
def load_articles(path: Path) -> list[dict]:
    return json.loads(path.read_text())
```

### 2. `build_dtm(texts)` and `build_tfidf(texts)`

Same pattern as Exercises 01–02:

```python
vectorizer = CountVectorizer(stop_words="english", min_df=2)
dtm = vectorizer.fit_transform(texts)
return dtm, vectorizer.get_feature_names_out()
```

### 3. `fit_lda(dtm, n_topics=4)`

Same as Exercise 01 - `random_state=42`, `max_iter=20`.

### 4. `top_words(model, feature_names, n=8)`

Return a list of word lists (one per topic), sorted by component weight.

### 5. `dominant_topic(model, dtm)`

```python
doc_topics = model.transform(dtm)
return [int(row.argmax()) for row in doc_topics]
```

### 6. `topic_purity(assignments, true_labels)`

Group documents by their assigned topic. For each topic, find the **majority category** - the true label that appears most often:

```python
purity = majority_count / total_in_topic
```

Return `[{"topic_id": 0, "majority_category": "sci.med", "count": 8, "total": 10, "purity": 0.8}, ...]`.

### 7. `contingency_matrix(assignments, true_labels, n_topics)`

Build a `n_topics x n_categories` count matrix:

```python
categories = sorted(set(true_labels))
matrix = [[0] * len(categories) for _ in range(n_topics)]
for topic_id, label in zip(assignments, true_labels):
    matrix[topic_id][cat_index[label]] += 1
```

Return `{"categories": [...], "matrix": [[...], ...]}`.

### 8. `topic_audit(records, n_topics=4)`

Full pipeline:

1. Build DTM from article texts
2. Fit LDA (k = `n_topics`)
3. Extract top words and dominant topics
4. Compute purity and contingency against `category` labels
5. Return structured report dict

---

## Run it

```bash
python module-04-topic-modelling/exercises/03-real-world-topics/start.py
```

## Run the tests

```bash
cd module-04-topic-modelling/exercises/03-real-world-topics
pytest test_start.py -v
```

---

## Optional hints

<details>
<summary>Hint: purity measures topic coherence against real labels</summary>

A topic with purity 1.0 contains only one category. Low purity means the topic mixes articles from different newsgroups - either `k` is wrong or the topics overlap semantically.

</details>

<details>
<summary>Hint: weighted average purity</summary>

Weight each topic's purity by its size (number of documents assigned) to get the overall average:

```python
avg = sum(p["purity"] * p["total"] for p in purity) / len(records)
```

</details>

<details>
<summary>Hint: contingency rows vs columns</summary>

Rows = discovered topic IDs (unsupervised). Columns = real categories (labels). A perfect model would have one non-zero entry per row and one per column.

</details>

<details>
<summary>Hint: don't use categories during training</summary>

`category` is for evaluation only. Feed only `text` to `CountVectorizer` and LDA. This simulates deploying unsupervised clustering on unlabelled data, then auditing against a sample with known labels.

</details>

<details>
<summary>Hint: why 20 Newsgroups?</summary>

It is the canonical text classification/clustering benchmark in sklearn. Four well-separated categories (sports, medicine, space, politics) let LDA recover clear themes - perfect for validating topic models before moving to messier real-world corpora.

</details>

---

## Checklist

- [ ] `load_articles` reads 40 records from JSON
- [ ] `build_dtm` uses `CountVectorizer(stop_words="english", min_df=2)`
- [ ] `fit_lda` uses `random_state=42`
- [ ] `topic_purity` computes per-topic majority category and purity
- [ ] `contingency_matrix` rows = topics, columns = categories, sums to 40
- [ ] `topic_audit` returns topics, purity, avg_purity, contingency, assignments
- [ ] `python start.py` prints readable audit report
- [ ] `pytest test_start.py -v` - all passed
