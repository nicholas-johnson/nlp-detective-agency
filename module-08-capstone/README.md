# Module 8 — Applied NLP Capstone

> The big case. A decade-old murder that went cold because the evidence was too scattered to connect — until now. Every tool you have built comes together: clean the files, vectorise the statements, classify the leads, extract the names, discover the hidden pattern, and crack it wide open.

## Learning goals

- Build an **end-to-end NLP pipeline** from raw text to actionable output.
- Compare a **classical baseline** (TF-IDF + classifier) against a **transformer** approach.
- **Persist models** with joblib and Hugging Face save/load for reuse.
- Expose a simple **inference API** so detectives can query the system.
- Run **evaluation and error analysis** to understand where the pipeline succeeds and fails.

---

## End-to-end pipeline — assembling the case

Chain preprocessing, feature extraction, classification, entity extraction, and summarisation into a single workflow that processes a new case file from intake to briefing.

<!-- Skeleton: pipeline architecture diagram, modular components, data flow from raw text to structured report -->

---

## Classical vs transformer — two detectives, one case

Run the same task with a TF-IDF + Logistic Regression baseline and a fine-tuned transformer. Compare accuracy, speed, and interpretability — know when each approach wins.

<!-- Skeleton: side-by-side evaluation, latency comparison, trade-off discussion -->

---

## Model persistence — filing the results

A model you cannot reload is a model you cannot deploy. Save sklearn pipelines with joblib and Hugging Face models with `save_pretrained` / `from_pretrained`.

<!-- Skeleton: joblib.dump/load, model.save_pretrained, pipeline bundling -->

---

## Inference API — the detective's dashboard

Wrap your pipeline in a simple FastAPI endpoint so detectives can submit a statement and get back classification, entities, and a summary without touching Python.

<!-- Skeleton: FastAPI endpoint, request/response schema, loading models at startup -->

---

## Evaluation and error analysis — the post-mortem

Aggregate metrics tell part of the story. Inspect misclassified documents, missed entities, and failed summaries to find the pipeline's weak spots and prioritise improvements.

<!-- Skeleton: error analysis notebook pattern, failure categorisation, iteration loop -->

---

## Field rules

- **Ship a baseline first.** A working classical pipeline beats a half-finished transformer.
- **Log inputs and outputs.** When the case goes to court, you need an audit trail.
- **Error analysis is not optional.** The capstone is not done until you know where it breaks.

---

## Exercises

| Folder | Mission |
| ------ | ------- |
| [`exercises/01-end-to-end-pipeline`](exercises/01-end-to-end-pipeline/) | Build the full pipeline: preprocess, classify, extract entities, summarise. |
| [`exercises/02-baseline-vs-transformer`](exercises/02-baseline-vs-transformer/) | Compare classical and transformer approaches on the big case dataset. |
| [`exercises/03-inference-api`](exercises/03-inference-api/) | Deploy the pipeline as a FastAPI service and query it with test case files. |

Run tests for this module:

```bash
pytest module-08-capstone/
```

## Slides

From repo root: `pnpm slides:08`, or `cd module-08-capstone/slides && pnpm dev`.

## Reference

- [joblib — persistence](https://joblib.readthedocs.io/en/latest/persistence.html)
- [Hugging Face — Saving and loading models](https://huggingface.co/docs/transformers/saving_models)
- [FastAPI](https://fastapi.tiangolo.com/)
