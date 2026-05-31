# Exercise 02 - Embedding Compass

Load a pre-trained GloVe model and run analogy and similarity queries that no Inkwell-trained Word2Vec model could answer - because "king", "paris", and "france" never appear in case files.

## Before you start

```bash
pip install -e ".[nlp,dev]"
python -c "import gensim.downloader; gensim.downloader.load('glove-wiki-gigaword-50')"
```

The download is ~66MB and happens once. Open `start.py` and implement each `# TODO`.

## What you'll build

```bash
python start.py
```

```
Embedding Compass - GloVe Report
========================================

Similarity:
  king / queen: 0.852
  man / woman: 0.832
  dock / pier: 0.612

Analogies:
  king - man + woman = queen
  paris - france + england = london

Odd one out from ['dock', 'pier', 'ledger', 'warehouse']: ledger
```

## Functions to implement

### 1. `load_pretrained(model_name="glove-wiki-gigaword-50")`

```python
import gensim.downloader as api
return api.load(model_name)
```

### 2. `similarity(model, word_a, word_b)`

```python
return float(model.similarity(word_a, word_b))
```

### 3. `analogy(model, pos, neg)`

```python
result = model.most_similar(positive=pos, negative=neg, topn=1)
return result[0][0]
```

### 4. `odd_one_out(model, words)`

```python
return model.doesnt_match(words)
```

### 5. `compass_report(model)`

Run these bundled queries:

**Similarity pairs:** (king, queen), (man, woman), (dock, pier)

**Analogies:**

- king - man + woman → ?
- paris - france + england → ?

**Odd one out:** [dock, pier, ledger, warehouse]

---

## Run it

```bash
python module-05-word-embeddings/exercises/02-embedding-compass/start.py
```

## Run the tests

```bash
cd module-05-word-embeddings/exercises/02-embedding-compass
pytest test_start.py -v
```

---

## Optional hints

<details>
<summary>Hint: KeyError on unknown words</summary>

GloVe only knows words from its training corpus. If a query word is missing, gensim raises `KeyError`. The bundled test words are all in GloVe.

</details>

<details>
<summary>Hint: analogy argument order</summary>

`most_similar(positive=["king", "woman"], negative=["man"])` computes king - man + woman.

</details>

---

## Checklist

- [ ] `load_pretrained` loads GloVe model
- [ ] `similarity` returns float in [-1, 1]
- [ ] `analogy` returns a string
- [ ] `odd_one_out` returns one word from the input list
- [ ] `compass_report` returns all three sections
- [ ] `pytest test_start.py -v` - all passed
