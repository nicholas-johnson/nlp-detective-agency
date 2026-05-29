# Module 4 — Topic Modelling

> Hundreds of cold cases sit in the archive, and somewhere in the pile, patterns are hiding. Cases that look unrelated may share the same motive, the same method, the same neighbourhood. Topic modelling discovers the recurring threads that connect seemingly unrelated files — and pins them to the evidence board.

## Learning goals

- Understand **unsupervised theme discovery** and when it beats manual categorisation.
- Apply **Latent Dirichlet Allocation (LDA)** to find latent topics in a document collection.
- Apply **Non-negative Matrix Factorization (NMF)** as an alternative topic model.
- Choose the **number of topics** using coherence scores and manual inspection.
- **Interpret and label** discovered topics, and **visualise** them with pyLDAvis.
- Summarise themes across a large corpus for investigative briefings.

---

## Unsupervised discovery — letting the archive speak

Nobody labelled these cold cases by theme. Topic modelling finds structure without labels — clusters of words that tend to co-occur across documents.

<!-- Skeleton: probabilistic vs matrix factorisation framing, document-topic and topic-word distributions -->

---

## Latent Dirichlet Allocation — the evidence clusters

LDA assumes every document is a mixture of topics, and every topic is a mixture of words. Run it on the archive and watch themes emerge.

<!-- Skeleton: LDA intuition, sklearn LatentDirichletAllocation, transform vs fit_transform -->

---

## Non-negative Matrix Factorization — the alternative lens

NMF factorises the document-term matrix into non-negative components. Often produces sharper, more interpretable topics than LDA on TF-IDF features.

<!-- Skeleton: NMF intuition, sklearn NMF, when to prefer NMF over LDA -->

---

## Choosing and interpreting topics — naming the threads

Too few topics and you miss nuance. Too many and you get noise. Coherence scores and manual review help you find the right number — then you label each topic for the evidence board.

<!-- Skeleton: coherence score overview, top words per topic, manual topic labelling, pyLDAvis -->

---

## Field rules

- **Topics are not categories.** They are soft mixtures — a document can belong to several themes.
- **Preprocessing matters enormously.** Stemming, stopwords, and min_df directly affect topic quality.
- **Always inspect top words manually.** Coherence scores guide you, but human judgement names the thread.

---

## Exercises

| Folder | Mission |
| ------ | ------- |
| [`exercises/01-lda-topics`](exercises/01-lda-topics/) | Run LDA on the cold-case archive and inspect discovered topics. |
| [`exercises/02-nmf-topics`](exercises/02-nmf-topics/) | Compare NMF topics with LDA on the same corpus. |
| [`exercises/03-topic-visualisation`](exercises/03-topic-visualisation/) | Visualise topics with pyLDAvis and produce an evidence-board summary. |

Run tests for this module:

```bash
pytest module-04-topic-modelling/
```

## Slides

From repo root: `pnpm slides:04`, or `cd module-04-topic-modelling/slides && pnpm dev`.

## Reference

- [scikit-learn — LatentDirichletAllocation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)
- [scikit-learn — NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html)
- [pyLDAvis](https://github.com/bmabey/pyLDAvis)
