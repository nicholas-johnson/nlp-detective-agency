# Exercise 03 - Semantic Search

> **Optional - requires an OpenAI API key.** Set `OPENAI_API_KEY` in your environment. Skip this exercise if you don't have one. Tests use mocked embeddings and run without a key.

Embed Inkwell witness statements with OpenAI and find semantically similar pairs that TF-IDF misses - because "the numbers guy" and "the accountant" share meaning, not vocabulary.

## Before you start

```bash
pip install -e ".[nlp,dev]"
export OPENAI_API_KEY=sk-...
```

Open `start.py`. Each function has a `# TODO` - implement them in order.

## The data

Witness statements in `data/inkwell/statements.json` - 10 records with `id`, `witness`, `raw_text`.

## What you'll build

```bash
python start.py
```

```
Inkwell Investigations - Semantic Search
============================================

Top pairs (OpenAI embeddings):
  STM-004 <-> STM-005  (0.812)
  ...

Top pairs (TF-IDF):
  STM-001 <-> STM-002  (0.245)
  ...

Pairs found by embeddings but not TF-IDF:
  STM-004 <-> STM-005
```

## Functions to implement

### 1. `load_statements(path)`

Read the JSON archive.

### 2. `embed_texts(texts, model="text-embedding-3-small")`

```python
from openai import OpenAI

client = OpenAI()
response = client.embeddings.create(input=texts, model=model)
return [item.embedding for item in response.data]
```

### 3. `cosine_pairs(embeddings, ids, n=5)`

Build cosine similarity matrix, extract top-n pairs from upper triangle.

### 4. `compare_with_tfidf(records, n=5)`

Return both embedding-based and TF-IDF-based top pairs:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

### 5. `search_report(records)`

Full workflow. Compute pairs found by embeddings but **not** in TF-IDF top-n - these are semantic matches word overlap missed.

---

## Run it

```bash
python module-05-word-embeddings/exercises/03-semantic-search/start.py
```

## Run the tests

Tests mock the API - no key needed:

```bash
cd module-05-word-embeddings/exercises/03-semantic-search
pytest test_start.py -v
```

---

## Optional hints

<details>
<summary>Hint: batch embedding</summary>

Pass all texts in one API call: `input=texts` accepts a list. Cheaper and faster than one call per statement.

</details>

<details>
<summary>Hint: why embeddings beat TF-IDF here</summary>

STM-004 mentions "accountant"; STM-005 mentions "numbers guy". No shared words - TF-IDF similarity is near zero. Embeddings capture that both describe a nervous financial witness.

</details>

---

## Checklist

- [ ] `load_statements` reads 10 records
- [ ] `embed_texts` calls OpenAI API
- [ ] `cosine_pairs` returns sorted top-n pairs
- [ ] `compare_with_tfidf` returns both pair lists
- [ ] `search_report` includes `only_in_embeddings`
- [ ] `pytest test_start.py -v` - all passed (mocked, no API key)
