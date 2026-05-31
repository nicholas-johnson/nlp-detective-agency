# Module 4 - Topic Modelling

> Hundreds of cold cases sit in the archive, and somewhere in the pile, patterns are hiding. Cases that look unrelated may share the same motive, the same method, the same neighbourhood. Topic modelling discovers the recurring threads that connect seemingly unrelated files - and pins them to the evidence board.

## Learning goals

- Understand **unsupervised theme discovery** and when it beats manual categorisation.
- Apply **Latent Dirichlet Allocation (LDA)** to find latent topics in a document collection.
- Apply **Non-negative Matrix Factorization (NMF)** as an alternative topic model.
- Choose the **number of topics** using perplexity and manual inspection of top words.
- **Interpret and label** discovered topics, and **visualise** them with pyLDAvis (demo).
- Summarise themes across a large corpus for investigative briefings.

## Setup

```bash
pip install -e ".[nlp,dev]"
```

This module uses **scikit-learn** for LDA and NMF. **pyLDAvis** is included for optional visualisation in the demo.

---

## Unsupervised discovery - letting the archive speak

Nobody labelled these cold cases by theme. **Topic modelling** finds structure without labels - clusters of words that tend to co-occur across documents. Unlike classification (Module 3), there are no correct answers and no single label per document.

A document can belong to **several topics at once** - a waterfront fraud case might mix dock vocabulary with ledger vocabulary. Topics are **soft mixtures**, not hard categories. A case file might be 70% "Waterfront" and 30% "Financial", reflecting language from both domains.

This makes topic modelling ideal for exploratory analysis: you discover themes you did not know to look for, rather than confirming categories you predefined.

---

## Document-term matrices - two vectorisers, two models

Before running any topic model, you need a numeric representation of the corpus. Module 2 introduced vectorisers; here the choice of vectoriser depends on the model:

| Model   | Vectoriser        | Why                                                            |
| ------- | ----------------- | -------------------------------------------------------------- |
| **LDA** | `CountVectorizer` | LDA models word **counts** - it expects integer frequencies    |
| **NMF** | `TfidfVectorizer` | NMF needs non-negative input; TF-IDF weights distinctive terms |

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

count_vec = CountVectorizer(stop_words="english", min_df=2)
dtm = count_vec.fit_transform(texts)

tfidf_vec = TfidfVectorizer(stop_words="english", min_df=2)
tfidf = tfidf_vec.fit_transform(texts)
```

### Why min_df and max_df matter

**`min_df=2`** drops words that appear in only one document. These rare terms are usually typos, proper nouns unique to a single case, OCR artefacts, or case-specific identifiers. They create **singleton topics** - topics dominated by one word from one document - that look meaningful but generalise to nothing.

**`max_df=0.9`** (or similar) drops words appearing in more than 90% of documents. These are function words the stopword list missed, or corpus-wide boilerplate ("said the detective", "according to the report"). They appear in every topic and add no discriminative structure.

Without these filters, topic quality degrades: you get topics like `{the, said, was, case, file}` or `{CASE-42}` instead of `{dock, warehouse, shipment, harbour}`.

**Preprocessing matters enormously.** Stopwords, min_df, and text quality directly affect whether topics are interpretable. Always inspect your vocabulary before running the model.

---

## Latent Dirichlet Allocation - the evidence clusters

Blei, Ng, and Jordan (2003) introduced **LDA** as a generative probabilistic model. The name describes its structure:

- **Latent** - topics are hidden variables, not observed in the data
- **Dirichlet** - prior distributions over topic mixtures are Dirichlet distributions
- **Allocation** - each word is allocated to one topic

### The generative story

LDA is a **generative model** — it describes a hypothetical random process that _could_ have produced the observed data. Nobody believes documents were literally generated this way; the model is a mathematical fiction. Its value is that when you invert the process (given the observed words, infer the hidden topics), you recover useful structure. This inversion is exactly what `fit_transform()` does.

LDA assumes documents were _generated_ by the following process:

1. For each document $d$, draw a **topic proportion vector** $\theta_d \sim \text{Dir}(\alpha)$
2. For each word position in document $d$:
   - Draw a **topic assignment** $z \sim \text{Multinomial}(\theta_d)$ — a multinomial distribution models the probability of picking one item from a fixed set of options, like rolling a loaded die where each face has a different probability; here, "draw a topic" means randomly selecting one topic according to the document's topic proportions
   - Draw a **word** $w \sim \text{Multinomial}(\phi_z)$ from that topic's word distribution

Topic modelling **inverts** this process: given the observed words, infer the hidden topic structure ($\theta$, $\phi$, and $z$).

### The Dirichlet distribution

The **Dirichlet distribution** is a "distribution over distributions". It generates probability vectors that sum to 1 - exactly what you need for topic proportions.

The concentration parameter $\alpha$ controls how mixed or focused documents are:

| $\alpha$ value          | Effect on documents                                         |
| ----------------------- | ----------------------------------------------------------- |
| $\alpha < 1$ (e.g. 0.1) | Sparse mixtures - each document is about few topics         |
| $\alpha = 1$            | Uniform — all possible topic mixtures are equally likely     |
| $\alpha > 1$ (e.g. 10)  | Uniform mixtures - every document blends all topics equally |

For investigative briefings, lower $\alpha$ often produces cleaner topic assignments - each case file maps to one or two dominant themes.

```python
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(
    n_components=4,
    random_state=42,
    max_iter=20,
)
doc_topics = lda.fit_transform(dtm)   # shape: (n_docs, n_topics)
```

- **`components_`** - topic-word distributions $\phi$ (shape: `n_topics × vocab_size`)
- **`fit_transform(dtm)`** - document-topic weights $\theta$ per file
- **`random_state=42`** - reproducible results for teaching

To inspect a topic, sort `components_[topic_id]` and read the highest-weight terms.

---

## Non-negative Matrix Factorization - the alternative lens

Lee and Seung (1999) introduced **NMF** as a parts-based factorisation. Where LDA uses probabilistic sampling, NMF uses linear algebra:

$$V \approx W \times H$$

| Matrix | Shape              | Meaning                    |
| ------ | ------------------ | -------------------------- |
| $V$    | documents × words  | Input document-term matrix |
| $W$    | documents × topics | Document-topic weights     |
| $H$    | topics × words     | Topic-word distributions   |

All entries are **non-negative**, which makes components directly interpretable as additive parts - a document _contains_ a weighted combination of topics, and each topic _contains_ a weighted combination of words. There are no negative weights to explain away.

On TF-IDF features, NMF often produces **sharper, more interpretable** topics than LDA because TF-IDF already down-weights common terms.

```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=4, random_state=42, max_iter=200)
doc_topics = nmf.fit_transform(tfidf)
```

Compare LDA and NMF top words side by side on the same corpus. Neither is universally better - inspect and choose for your briefing.

---

## Choosing the number of topics

Too few topics and you miss nuance. Too many and you get noise - topics that split a coherent theme into arbitrary sub-clusters.

### Perplexity

**Perplexity** measures how well the LDA model predicts held-in data. Lower perplexity means the model assigns higher probability to the words it sees:

$$\text{perplexity} = \exp\!\left(-\frac{\sum_{d=1}^{D} \log p(w_d)}{\sum_{d=1}^{D} N_d}\right)$$

where $p(w_d)$ is the model's probability of document $d$ and $N_d$ is its word count.

```python
for k in range(3, 7):
    lda = LatentDirichletAllocation(n_components=k, random_state=42)
    lda.fit(dtm)
    print(k, lda.perplexity(dtm))
```

**Warning:** perplexity often decreases monotonically as you add topics - more topics always give the model more flexibility to fit the data. Chang et al. (2009) showed that perplexity and human interpretability can **disagree**: the model with the lowest perplexity may produce topics that humans find incoherent. Always inspect top words manually.

### Topic coherence

**Topic coherence** measures whether the top words in a topic co-occur in real documents. Roder et al. (2015) proposed several variants ($C_V$, $C_{NPMI}$) that correlate better with human judgements than perplexity. A topic with words `{dock, harbour, shipment, warehouse}` coheres because those words appear together in actual text; `{dock, said, the, was}` does not.

This course uses perplexity plus human judgement to avoid extra dependencies, but coherence is the metric to reach for in production systems.

---

## Interpreting topics - naming the threads

Topic modelling output is not a finished briefing - it is raw material for human interpretation:

1. Print the **top 8–10 words** per topic from `components_` (LDA) or `H` (NMF)
2. Find each document's **dominant topic** - the index with highest weight
3. **Label topics manually** for the evidence board: "Waterfront", "Financial", "Surveillance", "Neighbourhood"
4. Group case IDs under each label for the briefing

```python
dominant = doc_topics.argmax(axis=1)
weight = doc_topics.max(axis=1)
```

Topics are hypotheses, not facts. A detective still validates every connection. Two topics with similar top words might need merging; one topic with mixed vocabulary might need splitting or more preprocessing.

---

## pyLDAvis - optional visual exploration

The demo can export an interactive HTML visualisation:

```python
import pyLDAvis
import pyLDAvis.lda_model

vis = pyLDAvis.lda_model.prepare(lda, dtm, count_vec)
pyLDAvis.save_html(vis, "archive_topics.html")
```

Open the HTML file in a browser to explore topic distances and word relevance. Exercises use structured Python output instead.

---

## Field rules

- **Topics are not categories.** They are soft mixtures - a document can belong to several themes.
- **Preprocessing matters enormously.** Stopwords, min_df, and text quality directly affect topic quality.
- **Always inspect top words manually.** Perplexity guides you, but human judgement names the thread.
- **Reproducibility:** set `random_state=42` on LDA and NMF.

---

## Demo

Interactive console menu - explore LDA, NMF, perplexity, and optional pyLDAvis on the cold-case archive:

```bash
python module-04-topic-modelling/demo/demo.py
```

---

## Exercises

| Folder                                                              | Mission                                                                                              |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| [`exercises/01-archive-themes`](exercises/01-archive-themes/)       | Run LDA on the cold-case archive and map each file to its dominant theme.                            |
| [`exercises/02-evidence-board`](exercises/02-evidence-board/)       | Compare LDA and NMF, pick the best topic count, and produce an evidence-board briefing.              |
| [`exercises/03-real-world-topics`](exercises/03-real-world-topics/) | Apply topic modelling to 20 Newsgroups articles and audit discovered topics against real categories. |

Run an exercise interactively:

```bash
python module-04-topic-modelling/exercises/01-archive-themes/start.py
python module-04-topic-modelling/exercises/02-evidence-board/start.py
python module-04-topic-modelling/exercises/03-real-world-topics/start.py
```

Run tests (from each exercise folder):

```bash
cd module-04-topic-modelling/exercises/01-archive-themes && pytest test_start.py -v
cd module-04-topic-modelling/exercises/02-evidence-board && pytest test_start.py -v
cd module-04-topic-modelling/exercises/03-real-world-topics && pytest test_start.py -v
```

## Slides

From repo root: `pnpm slides:04`, or `cd module-04-topic-modelling/slides && pnpm dev`.

## Reference

- [scikit-learn - LatentDirichletAllocation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
- [scikit-learn - NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
- [pyLDAvis](https://github.com/bmabey/pyLDAvis)
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet allocation. _Journal of Machine Learning Research_, 3, 993–1022.
- Lee, D. D., & Seung, H. S. (1999). Learning the parts of objects by non-negative matrix factorization. _Nature_, 401(6755), 788–791.
- Chang, J., Gerrish, S., Wang, C., Boyd-Graber, J. L., & Blei, D. M. (2009). Reading tea leaves: How humans interpret topic models. _NeurIPS_, 288–296.
- Roder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. _WSDM_, 399–408.
