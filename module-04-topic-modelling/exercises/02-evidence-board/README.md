# Exercise 02 - Evidence Board

Compare LDA and NMF on the same archive, pick the best topic count by perplexity, and produce an evidence-board briefing that groups cases under human-readable labels.

The demo shows an interactive perplexity table. This exercise **automates** model comparison and returns structured board groupings.

## Before you start

```bash
pip install -e ".[nlp,dev]"
```

Open `start.py`. Each function has a `# TODO` - implement them in order.

## The data

Same archive as Exercise 01: `data/inkwell/cold_cases.json` - 27 unlabeled case summaries.

## What you'll build

```bash
python start.py
```

prints best k, LDA vs NMF comparison, and evidence-board groupings:

```
Inkwell Investigations - Evidence Board
============================================

Best topic count (lowest perplexity): k=4

LDA vs NMF top words:
  Topic 0
    LDA: dock, warehouse, pier, fog, reeves, quay
    NMF: dock, pier, warehouse, fog, quay, reeves
...

Evidence board:
  Waterfront: ARC-001, ARC-002, ...
  Financial: ARC-008, ARC-009, ...
```

The `main()` function is already written.

## Functions to implement

### 1. `load_archive(path)`

Same as Exercise 01.

### 2. `perplexity_scores(dtm, k_values)`

```python
for k in k_values:
    lda = LatentDirichletAllocation(n_components=k, random_state=42, max_iter=20)
    lda.fit(dtm)
    perplexity = lda.perplexity(dtm)
```

Return `[{"k": 3, "perplexity": 412.5}, ...]` sorted by k. **Lower perplexity is better.**

### 3. `fit_nmf(tfidf_matrix, n_topics)`

```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=n_topics, random_state=42, max_iter=200)
nmf.fit(tfidf_matrix)
return nmf
```

### 4. `compare_models(lda_model, nmf_model, feature_names)`

Extract top-8 words per topic from both models:

```python
return {
    "lda": [["dock", "pier", ...], ...],
    "nmf": [["dock", "warehouse", ...], ...],
}
```

### 5. `label_topics(top_words)`

Map each topic to a human label using keyword rules:

| Label           | Keywords (examples)                              |
| --------------- | ------------------------------------------------ |
| Waterfront      | dock, pier, warehouse, quay, fog, reeves         |
| Financial       | ledger, accountant, office, receipt, station     |
| Surveillance    | motorcar, overcoat, plates, grey, mill           |
| Neighbourhood   | neighbour, shop, tram, tenant, orchard           |
| General         | fallback when no keywords match                  |

Count keyword overlaps in each topic's top words; pick the label with the highest score.

### 6. `evidence_board(records, k_values=None)`

Full workflow:

1. Build DTM and TF-IDF from summaries
2. Run `perplexity_scores` on k=3..6 (default); pick **lowest perplexity** k as `best_k`
3. Fit LDA on DTM and NMF on TF-IDF at `best_k`
4. Label topics and group case IDs by dominant LDA topic
5. Return `{best_k, lda_topics, nmf_topics, board: [{topic_label, case_ids}]}`

Every case ID must appear on the board exactly once.

---

## Run it

```bash
python module-04-topic-modelling/exercises/02-evidence-board/start.py
```

## Run the tests

```bash
cd module-04-topic-modelling/exercises/02-evidence-board
pytest test_start.py -v
```

---

## Optional hints

<details>
<summary>Hint: LDA vs NMF vectorisers</summary>

Use `CountVectorizer` for LDA perplexity and fitting. Use `TfidfVectorizer` for NMF - both with `stop_words="english", min_df=2`.

</details>

<details>
<summary>Hint: perplexity direction</summary>

Lower perplexity means the model assigns higher probability to the observed word counts. It is a guide, not ground truth - always read the top words for your chosen k.

</details>

<details>
<summary>Hint: LinearSVC-style dual not needed here</summary>

NMF only requires non-negative input. TF-IDF values are non-negative. Set `max_iter=200` to avoid convergence warnings on small corpora.

</details>

<details>
<summary>Hint: labelling is heuristic</summary>

Keyword rules are a teaching shortcut. In production, a detective labels topics after reading top words. `"General"` catches topics that do not match your keyword sets.

</details>

<details>
<summary>Hint: pyLDAvis is demo-only</summary>

The demo can export interactive HTML. This exercise returns structured Python dicts - no pyLDAvis import needed.

</details>

---

## Checklist

- [ ] `perplexity_scores` returns one entry per k value
- [ ] `fit_nmf` returns non-negative components
- [ ] `compare_models` has `lda` and `nmf` keys with top words
- [ ] `label_topics` returns string labels per topic_id
- [ ] `evidence_board` covers every case ID exactly once
- [ ] `python start.py` prints board groupings
- [ ] `pytest test_start.py -v` - all passed
