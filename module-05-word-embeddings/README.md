# Module 5 - Word Embeddings

> Suspects use aliases and code words. "The accountant", "numbers guy", and "the one with the ledger" might all point to the same person - but a Bag-of-Words model treats them as unrelated. Word embeddings build a map of meaning: words that appear in similar contexts land close together in vector space.

## Learning goals

- Understand **distributional semantics** - meaning from context.
- Train and use **Word2Vec** (CBOW and Skip-gram) embeddings with gensim.
- Load pre-trained **GloVe** vectors via `gensim.downloader`.
- Compute **word similarity**, **analogies**, and **embedding arithmetic**.
- Build **document embeddings** by averaging word vectors.
- Contrast **static** embeddings (Word2Vec) with **contextual** embeddings (transformers - Module 7).
- Use **API embeddings** optionally (OpenAI `text-embedding-3-small` - Exercise 03).

## Setup

```bash
pip install -e ".[nlp,dev]"
```

Download the pre-trained GloVe model once (needed for Exercise 02 and demo option 4):

```bash
python -c "import gensim.downloader; gensim.downloader.load('glove-wiki-gigaword-50')"
```

This module uses **gensim** for Word2Vec and pre-trained vectors, **scikit-learn** for cosine similarity and PCA.

---

## Distributional semantics - you shall know a word by the company it keeps

How do you know that "dock" and "pier" mean similar things? Not from their spelling - from the contexts where they appear. Firth (1957) captured this in a phrase that became the motto of computational linguistics: _"You shall know a word by the company it keeps."_

Harris (1954) formalised the idea: words with similar **distributional properties** (similar surrounding words) tend to have similar meanings. If `dock` and `pier` both appear near `waterfront`, `ship`, and `harbour`, a model can infer they are related - even without a dictionary.

**Word embeddings** turn this intuition into dense vectors. Each word maps to a point in $\mathbb{R}^d$ (typically $d = 50$ to $300$) such that semantically similar words are close together.

| Representation          | Shape                | Captures                                             |
| ----------------------- | -------------------- | ---------------------------------------------------- |
| BoW / TF-IDF (Module 2) | sparse, 10,000+ dims | Exact word identity - no similarity between synonyms |
| Word2Vec / GloVe        | dense, 50–300 dims   | Semantic similarity - synonyms cluster               |

Individual dimensions are **not interpretable** - no single axis means "waterfront" or "financial". Meaning is encoded in the **direction** of the vector across all dimensions together. That is why vector arithmetic works: the direction from `king` to `man` captures something like "royalty minus maleness", and adding it to `woman` lands near `queen`.

---

## Word2Vec - learning the map

Mikolov et al. (2013a) introduced **Word2Vec**, which trains a shallow neural network on a prediction task using co-occurrence statistics from a large corpus. The network has one hidden layer whose weight matrix _is_ the embedding table - each row is a word vector.

Two architectures share the same embedding matrix but predict differently:

| Model         | Predicts                             | Objective                                                                  | Best for                         |
| ------------- | ------------------------------------ | -------------------------------------------------------------------------- | -------------------------------- |
| **CBOW**      | Target word from surrounding context | $\max \log P(w_t \mid w_{t-k}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+k})$ | Faster; better on frequent words |
| **Skip-gram** | Context words from target word       | $\max \sum_{-k \leq j \leq k, j \neq 0} \log P(w_{t+j} \mid w_t)$          | Better on rare words             |

Both slide a **context window** of size $k$ across the corpus, creating training pairs from local co-occurrence patterns.

```python
from gensim.models import Word2Vec

sentences = [["dock", "warehouse", "reeves"], ["accountant", "ledger", "office"]]
model = Word2Vec(
    sentences,
    vector_size=50,
    window=5,
    min_count=2,
    workers=1,
    seed=42,
)
model.wv.most_similar("dock", topn=5)
```

- **`vector_size`** - dimensions per word (50 is fine for small corpora)
- **`window`** - how many neighbouring words to consider
- **`min_count`** - ignore words appearing fewer than N times

Train on Inkwell statements + cold-case summaries for domain-specific neighbours.

### Negative sampling

Computing the full softmax over a vocabulary of 100,000+ words at every training step is prohibitively expensive. Mikolov et al. (2013b) introduced **negative sampling**: instead of normalising over the entire vocabulary, contrast the true context word against a small number of randomly sampled "negative" words:

$$\log \sigma(\mathbf{v}_w \cdot \mathbf{v}_c) + \sum_{i=1}^{k} \log \sigma(-\mathbf{v}_w \cdot \mathbf{v}_{n_i})$$

where $\mathbf{v}_w$ is the target word vector, $\mathbf{v}_c$ is the true context word vector, $\mathbf{v}_{n_i}$ are negative sample vectors, and $\sigma$ is the sigmoid function. The model learns to pull co-occurring pairs together and push random pairs apart - efficiently approximating the full softmax.

---

## Pre-trained embeddings - standing on giants' shoulders

Training from scratch needs a large corpus. Pre-trained **GloVe** (Global Vectors) vectors were trained by Pennington, Socher, and Manning (2014) on billions of tokens from Wikipedia and Gigaword.

GloVe factorises the **word-word co-occurrence matrix** $X$, where $X_{ij}$ counts how often word $j$ appears in the context of word $i$. The training objective:

$$\sum_{i,j} f(X_{ij}) \left(\mathbf{w}_i^T \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2$$

The weighting function $f$ prevents very common co-occurrences from dominating. The key insight: ratios of co-occurrence probabilities encode meaning. $P(\text{ice} \mid \text{solid}) / P(\text{ice} \mid \text{gas})$ is large, capturing that ice relates to solid state; $P(\text{steam} \mid \text{gas}) / P(\text{steam} \mid \text{solid})$ captures the opposite.

```python
import gensim.downloader as api

model = api.load("glove-wiki-gigaword-50")  # 50-dimensional GloVe
model.similarity("king", "queen")
model.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
```

**Why pre-trained beats training on 37 Inkwell documents:** general vocabulary (king, paris, france) and stable analogies. Domain training wins for jargon (`reeves`, `quay`) that never appears in Wikipedia.

**Out-of-vocabulary (OOV):** pure Word2Vec returns nothing for unseen words. **fastText** (Bojanowski et al., 2017) handles this with character n-grams - `Reeves` gets a vector from subword pieces even if the full word was never seen during training.

---

## Similarity, analogies, and arithmetic

Once words are vectors, similarity is a geometric measurement. **Cosine similarity** (from Module 2) measures the angle between two vectors:

$$\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}$$

Scores near 1.0 mean the words appear in similar contexts; scores near 0 mean unrelated.

```python
# Cosine similarity
score = model.similarity("dock", "pier")

# Analogy: king - man + woman ≈ queen
result = model.most_similar(positive=["king", "woman"], negative=["man"], topn=1)

# Odd one out
model.doesnt_match(["dock", "pier", "ledger", "warehouse"])
```

Analogies work because consistent relational directions exist in the embedding space. If the vector difference $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}}$ captures "royalty minus gender", adding it to $\mathbf{v}_{\text{woman}}$ should land near $\mathbf{v}_{\text{queen}}$. This is approximate - not all analogies work perfectly - but it demonstrates that the space encodes relational structure, not just individual word meanings.

---

## Document embeddings by averaging

Individual word vectors capture word-level meaning. To represent an entire document, a simple baseline **averages** the vectors of all in-vocabulary words:

```python
import numpy as np

def document_vector(model, text):
    tokens = [t for t in text.lower().split() if t in model]
    if not tokens:
        return None
    return np.mean([model[t] for t in tokens], axis=0)
```

**Works well:** short, single-topic text (witness statements).  
**Fails when:** long mixed-topic documents - `bank` near `river` and `bank` near `money` average to a meaningless midpoint.

Transformers (Module 7) solve this with contextual embeddings - one vector per token that depends on surrounding words.

---

## Contextual vs static - the leap to transformers

|                        | Word2Vec / GloVe            | BERT / transformers       |
| ---------------------- | --------------------------- | ------------------------- |
| Vectors per word       | One, always                 | Different per sentence    |
| "bank" in "river bank" | Same vector                 | Different vectors         |
| Training               | Shallow NN on co-occurrence | Deep attention on context |
| Module                 | This one                    | Module 7                  |

Static embeddings are a strong baseline. Contextual embeddings are the modern standard - but you need to understand static vectors first.

---

## API embeddings (optional)

Production systems often call an embedding API instead of running models locally:

```python
from openai import OpenAI

client = OpenAI()
response = client.embeddings.create(
    input=["The accountant left the office at six."],
    model="text-embedding-3-small",
)
vector = response.data[0].embedding  # 1536 dimensions
```

One vector per document, no OOV, captures context. Requires an API key - see Exercise 03.

---

## Visualisation - projecting to 2D

High-dimensional vectors cannot be plotted directly. **Principal Component Analysis (PCA)** projects them to 2D by finding the directions of maximum variance:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
coords = pca.fit_transform(word_vectors)
for word, (x, y) in zip(words, coords):
    print(f"{word:15} ({x:.2f}, {y:.2f})")
```

Clusters of related terms become visible without matplotlib - print coordinates to the console. Note that 2D projections lose most of the structure; nearby points in 2D were close in high-D, but the reverse is not guaranteed.

---

## Field rules

- **Pre-trained embeddings are general; domain training is specific.** Train Word2Vec on case data when aliases and jargon matter.
- **OOV words are a real problem.** fastText handles subwords; pure Word2Vec does not.
- **Document vectors from word averaging are a baseline.** They work on short text; transformers do better on long mixed-topic documents.
- **Exercise 03 is optional.** It requires an OpenAI API key - skip if you don't have one.

---

## Demo

Interactive console menu - train Word2Vec, explore neighbours, run analogies, compare with GloVe:

```bash
python module-05-word-embeddings/demo/demo.py
```

---

## Exercises

| Folder                                                              | Mission                                                                      | API key?     |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ------------ |
| [`exercises/01-alias-map`](exercises/01-alias-map/)                 | Train Word2Vec on Inkwell case files and explore alias clusters.             | No           |
| [`exercises/02-embedding-compass`](exercises/02-embedding-compass/) | Load pre-trained GloVe and run similarity, analogy, and odd-one-out queries. | No           |
| [`exercises/03-semantic-search`](exercises/03-semantic-search/)     | Embed statements with OpenAI and find pairs TF-IDF misses.                   | **Optional** |

Exercises are **independent** - complete them in any order.

Run an exercise interactively:

```bash
python module-05-word-embeddings/exercises/01-alias-map/start.py
python module-05-word-embeddings/exercises/02-embedding-compass/start.py
python module-05-word-embeddings/exercises/03-semantic-search/start.py
```

Run tests (from each exercise folder):

```bash
cd module-05-word-embeddings/exercises/01-alias-map && pytest test_start.py -v
cd module-05-word-embeddings/exercises/02-embedding-compass && pytest test_start.py -v
cd module-05-word-embeddings/exercises/03-semantic-search && pytest test_start.py -v
```

## Slides

From repo root: `pnpm slides:05`, or `cd module-05-word-embeddings/slides && pnpm dev`.

## Reference

- [gensim - Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
- [gensim - Downloader API](https://radimrehurek.com/gensim/downloader.html)
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [OpenAI - Embeddings guide](https://platform.openai.com/docs/guides/embeddings)
- Firth, J. R. (1957). A synopsis of linguistic theory, 1930–1955. _Studies in Linguistic Analysis_, 1–32.
- Harris, Z. S. (1954). Distributional structure. _Word_, 10(2–3), 146–162.
- Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013a). Efficient estimation of word representations in vector space. _arXiv:1301.3781_.
- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013b). Distributed representations of words and phrases and their compositionality. _NeurIPS_, 3111–3119.
- Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. _EMNLP_, 1532–1543.
- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. _TACL_, 5, 135–146.
