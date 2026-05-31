# Exercise 01 - Alias Map

Build a Word2Vec model on Inkwell case files so the agency's jargon lands on the map. Explore which words cluster together as alias candidates.

## Before you start

```bash
pip install -e ".[nlp,dev]"
```

Open `start.py`. Each function has a `# TODO` - implement them in order.

## The data

Uses existing Inkwell files - no new dataset:

- `data/inkwell/statements.json` - 10 witness statements
- `data/inkwell/cold_cases.json` - 27 case summaries

Combined: **37 documents** for training.

## What you'll build

```bash
python start.py
```

prints an alias map for seed words:

```
Inkwell Investigations - Alias Map
========================================
Vocabulary: 142 words

dock: docks (0.89), pier (0.82), quay (0.71)
warehouse: dock (0.75), warehouse (0.68), ...
```

## Functions to implement

### 1. `load_corpus(statements_path, cold_cases_path)`

Load both JSON files. Return `raw_text` from statements plus `summary` from cold cases.

### 2. `tokenize_corpus(texts)`

```python
tokens = re.findall(r"[a-z]+", text.lower())
tokens = [t for t in tokens if len(t) >= 3]
```

Return a list of token lists (one per document).

### 3. `train_word2vec(sentences, ...)`

```python
from gensim.models import Word2Vec

return Word2Vec(
    sentences,
    vector_size=50,
    window=5,
    min_count=2,
    workers=1,
    seed=42,
)
```

### 4. `nearest_neighbours(model, word, n=5)`

Use `model.wv.most_similar(word, topn=n)`. Return `[]` if word not in vocabulary.

### 5. `alias_map(model, seed_words)`

For each seed, collect top-3 neighbours into a dict.

---

## Run it

```bash
python module-05-word-embeddings/exercises/01-alias-map/start.py
```

## Run the tests

```bash
cd module-05-word-embeddings/exercises/01-alias-map
pytest test_start.py -v
```

---

## Optional hints

<details>
<summary>Hint: min_count on small corpora</summary>

With only 37 documents, `min_count=2` drops very rare words. Lower to 1 if vocabulary is too small - but 2 reduces noise.

</details>

<details>
<summary>Hint: seed words not in vocabulary</summary>

If a seed word never appears twice, it won't be in the model. Return an empty list and handle gracefully in `main()`.

</details>

---

## Checklist

- [ ] `load_corpus` returns 37 texts
- [ ] `tokenize_corpus` returns list of token lists
- [ ] `train_word2vec` uses `seed=42`
- [ ] `nearest_neighbours` handles OOV words
- [ ] `alias_map` returns dict keyed by seed words
- [ ] `pytest test_start.py -v` - all passed
