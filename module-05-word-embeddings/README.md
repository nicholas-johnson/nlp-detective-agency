# Module 5 — Word Embeddings

> Suspects use aliases and code words. "The accountant", "numbers guy", and "the one with the ledger" might all point to the same person — but a Bag-of-Words model treats them as unrelated. Word embeddings build a map of meaning: words that appear in similar contexts land close together in vector space.

## Learning goals

- Understand **distributional semantics** — meaning from context.
- Train and use **Word2Vec** (CBOW and Skip-gram) embeddings.
- Load pre-trained **GloVe** and **fastText** vectors.
- Compute **word similarity**, **analogies**, and **embedding arithmetic**.
- **Visualise** embedding spaces with t-SNE and PCA.
- Use embeddings as **features** for downstream classification tasks.

---

## Distributional semantics — you shall know a word by the company it keeps

Words that appear in similar contexts tend to have similar meanings. Embeddings encode that intuition as dense vectors in high-dimensional space.

<!-- Skeleton: co-occurrence intuition, dense vs sparse representations, why embeddings beat BoW for semantics -->

---

## Word2Vec — learning the map

Word2Vec trains shallow neural networks to predict context from word (CBOW) or word from context (Skip-gram). The resulting vectors capture surprisingly rich semantic relationships.

<!-- Skeleton: CBOW vs Skip-gram, gensim Word2Vec API, training on domain corpus -->

---

## Pre-trained embeddings — standing on giants' shoulders

Training from scratch needs a large corpus. GloVe and fastText ship with pre-trained vectors trained on billions of tokens — ready to use on day one.

<!-- Skeleton: loading GloVe/fastText, spaCy vectors, out-of-vocabulary handling -->

---

## Similarity, analogies, and arithmetic — detective work in vector space

If "king" - "man" + "woman" ≈ "queen", then "suspect" - "alias" + "real_name" might reveal an identity. Explore similarity and analogy operations on your embedding map.

<!-- Skeleton: cosine similarity, most_similar, vector arithmetic examples -->

---

## Visualisation and downstream use — from map to evidence

Project embeddings to 2D with t-SNE or PCA to see clusters of related terms. Average document vectors to use embeddings as features in a classifier.

<!-- Skeleton: t-SNE/PCA plotting, document embedding by averaging, feeding embeddings into sklearn classifiers -->

---

## Field rules

- **Pre-trained embeddings are general; domain training is specific.** Fine-tune or train on case data when aliases and jargon matter.
- **OOV words are a real problem.** fastText handles subwords; pure Word2Vec does not.
- **Document vectors from word averaging are a baseline.** They work, but transformers do better.

---

## Exercises

| Folder | Mission |
| ------ | ------- |
| [`exercises/01-word2vec-training`](exercises/01-word2vec-training/) | Train Word2Vec on case-file vocabulary and explore nearest neighbours. |
| [`exercises/02-embedding-similarity`](exercises/02-embedding-similarity/) | Find alias clusters and run analogy queries on suspect code words. |
| [`exercises/03-embedding-classifier`](exercises/03-embedding-classifier/) | Use document embeddings as features for a classification task. |

Run tests for this module:

```bash
pytest module-05-word-embeddings/
```

## Slides

From repo root: `pnpm slides:05`, or `cd module-05-word-embeddings/slides && pnpm dev`.

## Reference

- [gensim — Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [spaCy — Vectors and similarity](https://spacy.io/usage/vectors-similarity)
