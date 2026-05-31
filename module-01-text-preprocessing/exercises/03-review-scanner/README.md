# Exercise 03 - Review Scanner

Exercise 02 built a preprocessing pipeline for Inkwell witness statements. Now apply the same techniques to **real movie reviews** from the NLTK corpus and see whether cleaned vocabulary separates positive from negative sentiment.

## Before you start

```bash
pip install -e ".[nlp,dev]"
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('movie_reviews')"
```

Open `start.py`. Each function has a `# TODO` - implement them in order.

## The data

Reviews live in `data/public/movie_reviews_sample.json`. Each record has:

| Field       | Description                |
| ----------- | -------------------------- |
| `id`        | Review ID (e.g. `REV-001`) |
| `sentiment` | `pos` or `neg`             |
| `text`      | Review excerpt             |

There are **40 reviews** - 20 positive, 20 negative - sampled from NLTK's `movie_reviews` corpus.

## What you'll build

```bash
python start.py
```

prints vocabulary stats and class-specific term rankings:

```
Movie Reviews - Preprocessing Audit
========================================
Vocabulary size: 412
Average tokens per review: 38.2

Top positive terms:
  film                 45
  ...

Most distinctive terms:
  awful                ratio=8.50  (neg)
```

The `main()` function is already written.

## Functions to implement

Reuse the Exercise 02 pipeline (`normalize` → `tokenize` → `stopwords` → `lemmatize`), adapted for review text (no `[REDACTED]` or `CASE-` patterns).

### 1. `load_reviews(path)`

### 2. `preprocess_review(text)`

Full pipeline; keep tokens with length >= 3.

### 3. `class_term_frequencies(records)`

Build separate `Counter` objects for `pos` and `neg` reviews.

### 4. `distinctive_terms(pos_freq, neg_freq, n=10)`

For each term, compute smoothed ratio `(count + 1)` between classes. Return the `n` terms with highest ratio and their dominant class.

### 5. `review_audit(records)`

Return `{pos_top, neg_top, distinctive, total_vocab, avg_tokens}`.

---

## Run it

```bash
python module-01-text-preprocessing/exercises/03-review-scanner/start.py
```

## Run the tests

```bash
cd module-01-text-preprocessing/exercises/03-review-scanner
pytest test_start.py -v
```

---

## Optional hints

<details>
<summary>Hint: smoothing ratios</summary>

Add 1 to each class count before dividing so rare terms do not divide by zero:

```python
pos = pos_freq.get(term, 0) + 1
neg = neg_freq.get(term, 0) + 1
```

</details>

<details>
<summary>Hint: distinctive vs frequent</summary>

`film` may be frequent in both classes. Distinctive terms have a **high ratio** in one class - they separate sentiment better than raw frequency.

</details>

---

## Checklist

- [ ] `load_reviews` reads 40 records
- [ ] `preprocess_review` applies full pipeline
- [ ] `class_term_frequencies` returns pos and neg Counters
- [ ] `distinctive_terms` returns ratio-ranked terms
- [ ] `review_audit` returns complete report dict
- [ ] `pytest test_start.py -v` - all passed
