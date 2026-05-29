# Module 2 — Feature Extraction

> The files are legible now. Time to dust them for prints — turn raw text into numerical signatures the system can compare, search, and cluster. Bag-of-Words, n-grams, and TF-IDF are the classical fingerprinting techniques that power most text analysis pipelines.

## Learning goals

- Represent documents as **Bag-of-Words** vectors and understand the document-term matrix.
- Capture word order with **n-grams** (unigrams, bigrams, trigrams).
- Apply **TF-IDF** weighting to emphasise distinctive terms over common ones.
- Use scikit-learn's **CountVectorizer** and **TfidfVectorizer** to vectorise a corpus.
- Tune **vocabulary size**, **min/max document frequency**, and **n-gram range** for your dataset.

## Setup

```bash
pip install -e ".[nlp,dev]"
```

This module uses **scikit-learn** for vectorisation. NLTK from Module 1 is not required here — the vectorisers handle lowercasing and English stopwords.

---

## Bag-of-Words — the word fingerprint

Ignore grammar, ignore order — just count. A Bag-of-Words representation turns every document into a vector of word frequencies, making text comparable with basic maths.

Given three short witness statements:

| Document | Text |
| -------- | ---- |
| D1 | the dock was empty |
| D2 | the warehouse was empty |
| D3 | the dock was busy |

The vocabulary is `{busy, dock, empty, warehouse, was}` and each document becomes a vector of counts:

```
           busy  dock  empty  warehouse  was
D1          0     1      1        0        1
D2          0     0      1        1        1
D3          1     1      0        0        1
```

In practice, vocabularies are **much larger** and matrices are **sparse** — most entries are zero because any one document only uses a fraction of all words.

```python
from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "the dock was empty",
    "the warehouse was empty",
    "the dock was busy",
]
vectorizer = CountVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(texts)

print(matrix.shape)                          # (3, 3) — 3 docs, 3 features
print(vectorizer.get_feature_names_out())    # ['busy' 'dock' 'empty' ...]
print(matrix.toarray())
```

---

## N-grams — catching phrases, not just words

"not guilty" means something very different from "guilty". N-grams capture short phrases that single words miss.

| N-gram type | Example from "near the dock" |
| ----------- | ---------------------------- |
| Unigram (1) | `near`, `the`, `dock` |
| Bigram (2)  | `near the`, `the dock` |
| Trigram (3) | `near the dock` |

Set the range with `ngram_range=(1, 2)` — includes both unigrams and bigrams:

```python
vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))
matrix = vectorizer.fit_transform(texts)
print(len(vectorizer.get_feature_names_out()))  # larger vocabulary
```

**Trade-off:** higher n captures more context but explodes vocabulary size and sparsity.

---

## TF-IDF — weighting what matters

Every case file mentions "the", "said", and "detective". Raw counts treat common words as important. **TF-IDF** down-weights terms that appear everywhere and highlights words distinctive to a specific document.

- **TF** (term frequency) — how often a term appears in *this* document.
- **IDF** (inverse document frequency) — penalises terms that appear in *many* documents.

$$\text{tf-idf}(t, d) = \text{tf}(t, d) \times \log\frac{N}{\text{df}(t)}$$

A word like `dock` that appears in several CASE-42 statements gets a lower IDF than a word unique to one witness.

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(texts)

# Highest TF-IDF terms in document 0
row = matrix[0].toarray().flatten()
names = vectorizer.get_feature_names_out()
top_idx = row.argsort()[-3:][::-1]
print([names[i] for i in top_idx])
```

---

## Vectorisers in scikit-learn — the fingerprint kit

Both `CountVectorizer` and `TfidfVectorizer` share the same interface:

| Method | Purpose |
| ------ | ------- |
| `fit(texts)` | Learn vocabulary from training documents |
| `transform(texts)` | Convert new documents to vectors using learned vocabulary |
| `fit_transform(texts)` | Both in one step |
| `get_feature_names_out()` | List of vocabulary terms |

**Key parameters:**

| Parameter | What it does |
| --------- | ------------ |
| `stop_words="english"` | Remove common function words |
| `lowercase=True` | Fold case (default) |
| `ngram_range=(1, 1)` | Unigrams only; `(1, 2)` adds bigrams |
| `min_df=2` | Ignore terms appearing in fewer than 2 documents |
| `max_df=0.9` | Ignore terms appearing in more than 90% of documents |
| `max_features=5000` | Cap vocabulary at the top N terms |

```python
vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1,
    max_features=1000,
)
matrix = vectorizer.fit_transform(train_texts)
test_matrix = vectorizer.transform(test_texts)  # same vocabulary
```

**Cosine similarity** compares TF-IDF vectors — useful for finding witness statements that tell similar stories:

```python
from sklearn.metrics.pairwise import cosine_similarity

sims = cosine_similarity(matrix)
# sims[i][j] = similarity between document i and j
```

---

## Field rules

- **Fit the vectoriser on training data only.** Leaking test vocabulary inflates your metrics.
- **High-dimensional sparse matrices are normal.** Do not densify them unless you have to.
- **Inspect your vocabulary.** Surprising tokens (typos, artefacts) often reveal preprocessing gaps.

---

## Demo

Interactive console menu — explore BoW, TF-IDF, n-grams, and similarity on Inkwell statements:

```bash
python module-02-feature-extraction/demo/demo.py
```

---

## Exercises

| Folder | Mission |
| ------ | ------- |
| [`exercises/01-document-fingerprints`](exercises/01-document-fingerprints/) | Build a BoW fingerprint card for every witness statement in a case. |
| [`exercises/02-matching-prints`](exercises/02-matching-prints/) | Use TF-IDF and cosine similarity to find the most similar witness pair. |

Run an exercise interactively:

```bash
python module-02-feature-extraction/exercises/01-document-fingerprints/start.py CASE-42
python module-02-feature-extraction/exercises/02-matching-prints/start.py CASE-42
```

Run tests (from each exercise folder):

```bash
cd module-02-feature-extraction/exercises/01-document-fingerprints && pytest test_start.py -v
cd module-02-feature-extraction/exercises/02-matching-prints && pytest test_start.py -v
```

## Slides

From repo root: `pnpm slides:02`, or `cd module-02-feature-extraction/slides && pnpm dev`.

## Reference

- [scikit-learn — CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [scikit-learn — TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn — Feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
