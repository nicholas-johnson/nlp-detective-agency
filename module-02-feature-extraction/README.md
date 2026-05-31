# Module 2 - Feature Extraction

> The files are legible now. Time to dust them for prints - turn raw text into numerical signatures the system can compare, search, and cluster. Bag-of-Words, n-grams, and TF-IDF are the classical fingerprinting techniques that power most text analysis pipelines.

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

This module uses **scikit-learn** for vectorisation. NLTK from Module 1 is not required here - the vectorisers handle lowercasing and English stopwords.

---

## The vector space model - documents as points in space

Before computers can compare two witness statements, text must become numbers. Salton, Wong, and Yang (1975) introduced the **vector space model (VSM)**: treat each document as a point in a high-dimensional space where each axis corresponds to one word in the vocabulary. Two documents that use similar words land close together; documents about different subjects land far apart.

The key insight is geometric: you do not need to understand grammar or meaning - you only need to count which words appear and compare the resulting vectors. Similarity becomes a mathematical operation (angle, distance) rather than a human judgement call.

This model underpins search engines, plagiarism detectors, and the similarity exercises in this module. Its limitation - ignoring word order - is exactly what n-grams (below) and later embeddings (Module 5) address.

---

## Bag-of-Words - the word fingerprint

The simplest VSM representation is **Bag-of-Words (BoW)**: ignore grammar, ignore order - just count how many times each word appears in each document.

Given three short witness statements:

| Document | Text                    |
| -------- | ----------------------- |
| D1       | the dock was empty      |
| D2       | the warehouse was empty |
| D3       | the dock was busy       |

After removing stopwords (`the`, `was`), the vocabulary is `{busy, dock, empty, warehouse}` and each document becomes a vector of counts:

```
           busy  dock  empty  warehouse
D1          0     1      1        0
D2          0     0      1        1
D3          1     1      0        0
```

This table is a **document-term matrix (DTM)** — the standard representation in text analysis. Rows are documents, columns are vocabulary terms, and cells hold counts (or weights, when using TF-IDF). Every text analysis technique in this course — similarity, classification, topic modelling — starts from a DTM.

D1 and D3 share the word `dock` but differ on `empty` vs `busy`. D2 is distinct because it mentions `warehouse` instead of `dock`. A classifier (Module 3) or similarity search can use these differences directly.

In practice, vocabularies are **much larger** — thousands or tens of thousands of terms — and matrices are **sparse**: most entries are zero because any one document uses only a tiny fraction of all words.

```python
from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "the dock was empty",
    "the warehouse was empty",
    "the dock was busy",
]
vectorizer = CountVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(texts)

print(matrix.shape)                          # (3, 3) - 3 docs, 3 features
print(vectorizer.get_feature_names_out())    # ['busy' 'dock' 'empty' ...]
print(matrix.toarray())
```

---

## N-grams - catching phrases, not just words

"not guilty" means something very different from "guilty". Single-word BoW loses negation, compound nouns, and idioms. **N-grams** capture short contiguous sequences of tokens:

| N-gram type | Example from "near the dock" |
| ----------- | ---------------------------- |
| Unigram (1) | `near`, `the`, `dock`        |
| Bigram (2)  | `near the`, `the dock`       |
| Trigram (3) | `near the dock`              |

Set the range with `ngram_range=(1, 2)` - includes both unigrams and bigrams:

```python
vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))
matrix = vectorizer.fit_transform(texts)
print(len(vectorizer.get_feature_names_out()))  # larger vocabulary
```

**Trade-off:** higher n captures more context but explodes vocabulary size and sparsity. A corpus with 5,000 unique unigrams might produce 50,000+ bigrams. Use bigrams when phrase boundaries matter (sentiment, legal language); stick to unigrams for large, noisy corpora.

---

## TF-IDF - weighting what matters

Every case file mentions common words. Raw counts treat a word that appears in every document as equally important as a word unique to one witness. **TF-IDF** (term frequency–inverse document frequency) down-weights terms that appear everywhere and up-weights terms distinctive to a specific document.

Sparck Jones (1972) introduced the intuition that rare terms are more informative than common ones. The classic formula combines two components:

$$\text{tf-idf}(t, d) = \text{tf}(t, d) \times \text{idf}(t)$$

where:

$$\text{idf}(t) = \log \frac{N}{\text{df}(t)}$$

- **TF** (term frequency) - how often term $t$ appears in document $d$
- **IDF** (inverse document frequency) - $\text{df}(t)$ is the number of documents containing term $t$; $N$ is the total number of documents
- The **log** dampens the effect of very common terms - without it, a word appearing in 99% of documents would get an IDF near zero, but the ratio would still dominate numerically

### TF variants

scikit-learn's `TfidfVectorizer` supports several TF normalisations:

| Variant    | Formula                                        | Effect                       |
| ---------- | ---------------------------------------------- | ---------------------------- |
| Raw count  | $\text{tf}(t,d) = \text{count}(t,d)$           | Long documents dominate      |
| Log-scaled | $\text{tf}(t,d) = 1 + \log(\text{count}(t,d))$ | Dampens repeated words       |
| Boolean    | $\text{tf}(t,d) \in \{0, 1\}$                  | Presence only, not frequency |

### IDF smoothing

scikit-learn uses smoothed IDF to avoid division by zero for terms that appear in every document:

$$\text{idf}(t) = \log \frac{1 + N}{1 + \text{df}(t)} + 1$$

After computing TF-IDF, scikit-learn **L2-normalises** each document vector so that longer documents do not automatically get higher scores just because they contain more words.

A word like `dock` that appears in several CASE-42 statements gets a lower IDF than a word unique to one witness - making the unique word a stronger fingerprint.

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

## Cosine similarity - comparing fingerprints

Once documents are vectors, you need a way to measure how similar they are. **Cosine similarity** measures the angle between two vectors, ignoring their magnitude (length):

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|} = \frac{\sum_i a_i b_i}{\sqrt{\sum_i a_i^2} \, \sqrt{\sum_i b_i^2}}$$

Scores range from 0 (orthogonal - no shared terms) to 1 (identical direction).

**Why cosine over Euclidean distance?** Euclidean distance is sensitive to document length. A 500-word witness statement and a 50-word summary of the same event would be far apart in Euclidean space simply because the long document has larger counts. Cosine similarity ignores length - it cares only about the _proportion_ of each word, so two documents about the same topic score highly even when one is much longer.

**Worked example** on the three-document toy corpus (after stopword removal):

|     | busy | dock | empty | warehouse |
| --- | ---- | ---- | ----- | --------- |
| D1  | 0    | 1    | 1     | 0         |
| D3  | 1    | 1    | 0     | 0         |

$$\cos(D1, D3) = \frac{0 \cdot 1 + 1 \cdot 1 + 1 \cdot 0 + 0 \cdot 0}{\sqrt{2} \cdot \sqrt{2}} = \frac{1}{2} = 0.5$$

D1 and D3 share `dock` but differ on `empty` vs `busy` - moderate similarity. D1 and D2 share only `empty` - lower similarity.

```python
from sklearn.metrics.pairwise import cosine_similarity

sims = cosine_similarity(matrix)
# sims[i][j] = similarity between document i and j
```

---

## Sparsity - why text matrices are special

Text feature matrices are **sparse**: the vast majority of entries are zero. A corpus with 10,000 unique words and 1,000 documents has 10 million matrix cells, but each document might use only 100–300 distinct words - meaning 97–99% of entries are zero.

scikit-learn stores sparse matrices in **compressed sparse row (CSR)** format internally. Instead of allocating a cell for every document-word pair, CSR stores only the non-zero values and their row/column positions. A matrix that is 97% zeros uses roughly 3% of the memory a full dense array would need, and arithmetic operations skip the zeros entirely. This is why scikit-learn vectorisers return sparse matrices by default.

You should **not** call `.toarray()` on large corpora — converting a 10,000 × 50,000 sparse matrix to dense would require gigabytes of RAM.

```python
matrix = vectorizer.fit_transform(texts)
print(type(matrix))          # scipy.sparse matrix
print(matrix.nnz)            # number of non-zero entries
# matrix.toarray()           # only for tiny toy examples
```

When inspecting results, use `.toarray()` only on small subsets or individual rows.

---

## Vectorisers in scikit-learn - the fingerprint kit

Both `CountVectorizer` and `TfidfVectorizer` share the same interface:

| Method                    | Purpose                                                   |
| ------------------------- | --------------------------------------------------------- |
| `fit(texts)`              | Learn vocabulary from training documents                  |
| `transform(texts)`        | Convert new documents to vectors using learned vocabulary |
| `fit_transform(texts)`    | Both in one step                                          |
| `get_feature_names_out()` | List of vocabulary terms                                  |

**Key parameters:**

| Parameter              | What it does                                         |
| ---------------------- | ---------------------------------------------------- |
| `stop_words="english"` | Remove common function words                         |
| `lowercase=True`       | Fold case (default)                                  |
| `ngram_range=(1, 1)`   | Unigrams only; `(1, 2)` adds bigrams                 |
| `min_df=2`             | Ignore terms appearing in fewer than 2 documents     |
| `max_df=0.9`           | Ignore terms appearing in more than 90% of documents |
| `max_features=5000`    | Cap vocabulary at the top N terms                    |

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

**Critical rule:** `fit` on training data only. `transform` on test data using the vocabulary learned during `fit`. Fitting on the full corpus leaks test vocabulary into your model and inflates evaluation metrics.

---

## Field rules

- **Fit the vectoriser on training data only.** Leaking test vocabulary inflates your metrics.
- **High-dimensional sparse matrices are normal.** Do not densify them unless you have to.
- **Inspect your vocabulary.** Surprising tokens (typos, artefacts) often reveal preprocessing gaps.

---

## Demo

Interactive console menu - explore BoW, TF-IDF, n-grams, and similarity on Inkwell statements:

```bash
python module-02-feature-extraction/demo/demo.py
```

---

## Exercises

| Folder                                                                      | Mission                                                                             |
| --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| [`exercises/01-document-fingerprints`](exercises/01-document-fingerprints/) | Build a BoW fingerprint card for every witness statement in a case.                 |
| [`exercises/02-matching-prints`](exercises/02-matching-prints/)             | Use TF-IDF and cosine similarity to find the most similar witness pair.             |
| [`exercises/03-article-matcher`](exercises/03-article-matcher/)             | Apply TF-IDF similarity to real newsgroup articles and validate against categories. |

Run an exercise interactively:

```bash
python module-02-feature-extraction/exercises/01-document-fingerprints/start.py CASE-42
python module-02-feature-extraction/exercises/02-matching-prints/start.py CASE-42
python module-02-feature-extraction/exercises/03-article-matcher/start.py
```

Run tests (from each exercise folder):

```bash
cd module-02-feature-extraction/exercises/01-document-fingerprints && pytest test_start.py -v
cd module-02-feature-extraction/exercises/02-matching-prints && pytest test_start.py -v
cd module-02-feature-extraction/exercises/03-article-matcher && pytest test_start.py -v
```

## Slides

From repo root: `pnpm slides:02`, or `cd module-02-feature-extraction/slides && pnpm dev`.

## Reference

- [scikit-learn - CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [scikit-learn - TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn - Feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- Salton, G., Wong, A., & Yang, C. S. (1975). A vector space model for automatic indexing. _Communications of the ACM_, 18(11), 613–620.
- Sparck Jones, K. (1972). A statistical interpretation of term specificity and its application in retrieval. _Journal of Documentation_, 28(1), 11–21.
