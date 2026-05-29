# Module 7 — Transformers and Pre-trained Models

> The cold cases are too complex for the old methods. Witnesses contradict each other, motives are buried in subtext, and aliases fold into aliases. Time to call in the specialist — a pre-trained transformer model that reads context the way a seasoned detective does, catching nuance that bag-of-words and static embeddings miss.

## Learning goals

- Understand **attention** and the **transformer architecture** at a high level.
- Contrast **contextual embeddings** (BERT) with **static embeddings** (Word2Vec).
- Use Hugging Face **`pipeline`** for classification, NER, QA, and summarisation out of the box.
- Work with **tokenizers** — subword tokenisation, truncation, and padding.
- **Fine-tune** a pre-trained model with `AutoModel` and `AutoTokenizer` on agency data.

---

## Transformers and attention — how the specialist thinks

Traditional models read words in isolation. Transformers use self-attention to weigh every word against every other word in context — "bank" near "river" vs "bank" near "money" resolves differently.

<!-- Skeleton: attention intuition, encoder architecture, why context matters for NLP -->

---

## BERT and contextual embeddings — words in context

BERT produces different vectors for the same word in different sentences. This is the leap from "the accountant" always mapping to one vector, to mapping differently when the context changes.

<!-- Skeleton: masked language modelling, [CLS] token, BERT variants (distilbert, roberta overview) -->

---

## Hugging Face pipelines — deploy in minutes

The `pipeline` API wraps model, tokenizer, and post-processing into a single call. Classify statement tone, extract entities, answer questions about a case file, or summarise a long report — all in a few lines.

<!-- Skeleton: pipeline("sentiment-analysis"), pipeline("ner"), pipeline("question-answering"), pipeline("summarization") -->

---

## Tokenizers — splitting text the transformer way

Transformers use subword tokenisation (WordPiece, BPE). Learn to tokenise, truncate, pad, and decode so your inputs match what the model expects.

<!-- Skeleton: AutoTokenizer, encode/decode, max_length, truncation, padding, return_tensors -->

---

## Fine-tuning — training on the agency archives

Pre-trained models are generalists. Fine-tune on Inkwell's labelled case data to specialise in your domain — better sentiment on witness statements, sharper NER on case-specific entities.

<!-- Skeleton: Trainer API overview, training arguments, evaluation loop, saving/loading fine-tuned models -->

---

## Field rules

- **Start with a pipeline before you fine-tune.** Baseline quality first, then improve.
- **Fine-tuning needs labelled data and compute.** DistilBERT is a good starting point for speed.
- **Watch token limits.** Long case files need chunking or summarisation before feeding to the model.

---

## Exercises

| Folder | Mission |
| ------ | ------- |
| [`exercises/01-hf-pipelines`](exercises/01-hf-pipelines/) | Run pre-built pipelines for sentiment, NER, and summarisation on case files. |
| [`exercises/02-tokenization`](exercises/02-tokenization/) | Tokenise and prepare witness statements for transformer input. |
| [`exercises/03-fine-tuning`](exercises/03-fine-tuning/) | Fine-tune a DistilBERT model on labelled agency data. |

Run tests for this module:

```bash
pytest module-07-transformers/
```

## Slides

From repo root: `pnpm slides:07`, or `cd module-07-transformers/slides && pnpm dev`.

## Reference

- [Hugging Face — Transformers documentation](https://huggingface.co/docs/transformers/index)
- [Hugging Face — Pipelines](https://huggingface.co/docs/transformers/pipeline_tutorial)
- [Hugging Face — Fine-tuning](https://huggingface.co/docs/transformers/training)
- [Attention Is All You Need (paper)](https://arxiv.org/abs/1706.03762)
