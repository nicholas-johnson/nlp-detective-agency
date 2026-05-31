# Exercise 03 - Article Matcher

Exercise 02 found the most similar witness pair within a case using TF-IDF. Now run the same similarity workflow on **real newsgroup articles** and check whether high-similarity pairs share the same editorial category.

## Before you start

```bash
pip install -e ".[nlp,dev]"
```

Open `start.py`. Each function has a `# TODO` - implement them in order.

## The data

Articles live in `data/public/newsgroups_articles.json`. Each record has:

| Field      | Description                              |
| ---------- | ---------------------------------------- |
| `id`       | Article ID (e.g. `ART-001`)              |
| `category` | True newsgroup name (for evaluation)     |
| `text`     | Article body                             |

**40 articles** - 10 each from `comp.graphics`, `rec.autos`, `sci.electronics`, `soc.religion.christian`.

## What you'll build

```bash
python start.py
```

prints top similar pairs and how often they share a category:

```
20 Newsgroups - Similarity Audit
========================================
Articles: 40
Same-category rate (top pairs): 80%

Top similar pairs:
  ART-003 <-> ART-007  (0.412)
```

## Functions to implement

Reuse patterns from Exercise 02 (`build_tfidf_matrix`, `compare_ngram_vocab_sizes`).

### 1. `top_similar_pairs(sim_matrix, ids, n=5)`

Iterate the upper triangle of the similarity matrix. Return the top `n` pairs sorted by score.

### 2. `same_category_rate(pairs, records)`

What fraction of the top pairs have matching `category` values? This validates whether TF-IDF similarity recovers real topic structure.

### 3. `article_audit(records, n_pairs=5)`

Full workflow returning `{total_articles, vocab_sizes, top_pairs, same_category_rate}`.

---

## Run it

```bash
python module-02-feature-extraction/exercises/03-article-matcher/start.py
```

## Run the tests

```bash
cd module-02-feature-extraction/exercises/03-article-matcher
pytest test_start.py -v
```

---

## Optional hints

<details>
<summary>Hint: skip the diagonal</summary>

When comparing document `i` to `j`, only consider `j > i` to avoid duplicate pairs and self-comparisons (always 1.0).

</details>

<details>
<summary>Hint: same-category rate</summary>

A rate of 1.0 means every top pair shares a newsgroup. Lower rates suggest cross-topic similarity - useful for finding related but distinct themes.

</details>

---

## Checklist

- [ ] `load_articles` reads 40 records
- [ ] `top_similar_pairs` returns n pairs sorted by score
- [ ] `same_category_rate` returns float in [0, 1]
- [ ] `article_audit` returns complete report
- [ ] `pytest test_start.py -v` - all passed
