# Exercise 02 - Matching Prints

Two witnesses may be telling the same story. TF-IDF weights distinctive terms, and cosine similarity measures how close two statements are in meaning-space. Your job: find the most similar witness pair in a case and compare n-gram vocabulary sizes.

This builds on Exercise 01 concepts but uses **TfidfVectorizer** and **cosine similarity** - a different deliverable.

## Before you start

Complete Exercise 01 first (or read its README for `texts_for_case` patterns). You reimplement `load_statements` and `texts_for_case` here - exercises are self-contained.

```bash
pip install -e ".[nlp,dev]"
```

## The data

Same archive: `data/inkwell/statements.json`. `CASE-42` has 4 statements about the docks case.

## What you'll build

```bash
python start.py CASE-42
```

prints:

```
Inkwell Investigations - Similarity Report for CASE-42
============================================================

Most similar pair: STM-001 <-> STM-002  (score: 0.142)

Vocabulary sizes:
  Unigrams only:  87
  With bigrams:   142
```

(Exact similarity score and vocab sizes will vary slightly.)

## Functions to implement

### 1. `load_statements(path)` and `texts_for_case(statements, case_id)`

Same as Exercise 01 - copy the logic.

### 2. `build_tfidf_matrix(texts)`

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(texts)
return matrix, vectorizer.get_feature_names_out()
```

Same interface as `CountVectorizer`, but weights are TF-IDF scores (floats), not raw counts.

### 3. `distinctive_terms(matrix, feature_names, doc_index, n=5)`

Like `top_terms` in Exercise 01, but sort by TF-IDF **weight** (float), not count:

```python
row = matrix[doc_index].toarray().flatten()
ranked = row.argsort()[::-1]
return [(feature_names[i], round(float(row[i]), 3)) for i in ranked if row[i] > 0][:n]
```

### 4. `most_similar_pair(matrix, doc_ids)`

Find the pair of **different** documents with the highest cosine similarity:

```python
from sklearn.metrics.pairwise import cosine_similarity

sims = cosine_similarity(matrix)   # shape (n, n)
# sims[i][i] == 1.0 - skip the diagonal
# check all pairs where j > i, return (doc_ids[i], doc_ids[j], score)
```

Return `None` if fewer than 2 documents.

Cosine similarity ranges from 0 (no overlap) to 1 (identical vectors).

### 5. `compare_ngram_vocab_sizes(texts)`

Compare vocabulary size with and without bigrams:

```python
uni = CountVectorizer(stop_words="english", ngram_range=(1, 1))
bi = CountVectorizer(stop_words="english", ngram_range=(1, 2))
uni.fit(texts)
bi.fit(texts)
return {
    "unigram": len(uni.get_feature_names_out()),
    "bigram": len(bi.get_feature_names_out()),
}
```

Bigram vocabulary should be **larger** - it includes unigrams plus two-word phrases.

### 6. `similarity_report(statements, case_id)`

Combine into one report dict:

```python
{
    "case_id": case_id,
    "most_similar": (id_a, id_b, score) or None,
    "ngram_vocab_sizes": {"unigram": int, "bigram": int},
}
```

Return `most_similar: None` and zero vocab sizes for unknown cases.

---

## Run it

```bash
python module-02-feature-extraction/exercises/02-matching-prints/start.py CASE-42
```

## Run the tests

```bash
cd module-02-feature-extraction/exercises/02-matching-prints
pytest test_start.py -v
```

---

## Optional hints

<details>
<summary>Hint: skip the diagonal</summary>

`cosine_similarity` of a document with itself is always 1.0. When finding the _most similar pair_, only compare **different** documents:

```python
for i in range(len(doc_ids)):
    for j in range(i + 1, len(doc_ids)):
        score = sims[i][j]
```

</details>

<details>
<summary>Hint: TF-IDF vs raw counts</summary>

Exercise 01 used counts - "dock" appearing 3 times scores high. TF-IDF down-weights terms that appear in many documents. A word unique to one witness gets a higher TF-IDF score than a word all witnesses mention.

</details>

<details>
<summary>Hint: why bigram vocab is larger</summary>

`ngram_range=(1, 2)` captures both single words (`dock`) and phrases (`near dock`, `dock workers`). The vocabulary is a superset of unigrams alone, so the count is always >= unigram-only.

</details>

<details>
<summary>Hint: difference from the demo</summary>

The demo lets you **pick** two statements to compare. This exercise automatically finds the **highest-similarity pair** across all statements in a case - no user input needed.

</details>

---

## How this differs from Exercise 01

|            | Exercise 01                    | Exercise 02                      |
| ---------- | ------------------------------ | -------------------------------- |
| Vectoriser | CountVectorizer (BoW)          | TfidfVectorizer                  |
| Output     | Fingerprint card per statement | Most similar pair + n-gram stats |
| Metric     | Raw word counts                | Cosine similarity                |
| Scope      | All statements in a case       | Pairwise comparison              |

---

## Checklist

- [ ] `build_tfidf_matrix` uses `TfidfVectorizer(stop_words="english")`
- [ ] `distinctive_terms` returns TF-IDF weights, not counts
- [ ] `most_similar_pair` skips diagonal, returns highest off-diagonal pair
- [ ] `compare_ngram_vocab_sizes` returns unigram and bigram vocab sizes
- [ ] `similarity_report` handles unknown cases gracefully
- [ ] `python start.py CASE-42` prints the report
- [ ] `pytest test_start.py -v` - all passed
