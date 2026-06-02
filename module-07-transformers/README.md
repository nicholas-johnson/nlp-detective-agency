# Module 7 - Transformers and Pre-trained Models

> The cold cases are too complex for the old methods. Witnesses contradict each other, motives are buried in subtext, and aliases fold into aliases. Time to call in the specialist - a pre-trained transformer model that reads context the way a seasoned detective does, catching nuance that bag-of-words and static embeddings miss.

## Learning goals

- Understand **attention** and the **transformer architecture** at a high level.
- Contrast **contextual embeddings** (BERT) with **static embeddings** (Word2Vec).
- Use Hugging Face **`pipeline`** for classification, NER, and zero-shot inference.
- Work with **tokenizers** - BPE via tiktoken and WordPiece via AutoTokenizer.
- **Fine-tune** a pre-trained model optionally (DistilBERT on agency data).

## Setup

**Exercises 01 and 02** (pipelines + tokenization):

```bash
pip install -e ".[nlp,local-ml,dev]"
```

HF `pipeline` inference requires **PyTorch** (`torch` in `[local-ml]`). First run downloads models (~250 MB, cached by Hugging Face).

**Exercise 03 - Fine-tuning (optional):**

Same install - `[local-ml]` includes `torch`, `transformers`, `datasets`, and `accelerate`. Skip Ex03 if you lack GPU time or disk space.

**Exercise 02 tokenization** uses **tiktoken** (already in core deps) - no model download needed for the main tasks. AutoTokenizer comparison loads DistilBERT only when you call `compare_tokenizers()`.

---

## Attention and transformers - how the specialist thinks

Traditional models (BoW, Word2Vec) read words in isolation or from a fixed context window. **Transformers** (Vaswani et al., 2017) process all tokens simultaneously and use **self-attention** to weigh every token against every other token in the input. The "self" means the model attends within the _same_ input sequence — each token looks at every other token in that sentence to decide what context matters. This contrasts with cross-attention (used in translation models), where tokens in one language attend to tokens in another.

"Bank" near "river" and "bank" near "money" resolve differently because the model attends to different surrounding words - not because someone hand-coded the distinction.

```
Input tokens → Embedding layer → Multi-head self-attention (×N layers) → Task head
```

A **task head** is a small output layer added on top of the transformer's general-purpose representation. For classification, it maps the final hidden state to class probabilities. For NER, it maps each token's hidden state to an entity tag. Swapping the head lets the same pre-trained transformer serve different tasks without retraining the whole model.

You do not need to implement attention — understand that **context flows through the stack** and the same word gets a different representation in different sentences.

### Scaled dot-product attention

Each token produces three vectors: a **query** (what am I looking for?), a **key** (what do I contain?), and a **value** (what information do I pass forward?). Attention scores measure how much each token should attend to every other token:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- $QK^T$ computes pairwise relevance scores between all tokens
- Dividing by $\sqrt{d_k}$ (the key dimension) prevents dot products from growing large, which would push softmax into saturated regions with tiny gradients
- Softmax normalises scores into weights that sum to 1
- Multiplying by $V$ produces a weighted sum of value vectors - each token's output is a blend of all tokens, weighted by relevance

### Multi-head attention

Instead of one attention pass, transformers run $h$ parallel **attention heads**, each with its own learned projections for $Q$, $K$, and $V$. Heads are concatenated and projected back:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O$$

Different heads can specialise - one might track syntactic relationships, another semantic similarity, another positional proximity. BERT-base uses 12 heads; DistilBERT uses 6.

### Positional encodings

Unlike RNNs, transformers process all tokens in parallel - they have no inherent notion of word order. **Positional encodings** are added to token embeddings so the model knows that "dog bites man" and "man bites dog" are different.

The original transformer used fixed sinusoidal encodings. BERT and DistilBERT use **learned positional embeddings** - a lookup table indexed by position (0, 1, 2, …). Without positional information, word order would be invisible to the model.

---

## BERT and contextual embeddings - words in context

Devlin et al. (2019) introduced **BERT** (Bidirectional Encoder Representations from Transformers). Unlike Word2Vec (Module 5), BERT produces a **different vector for the same word in different sentences** - because the representation is built from the full surrounding context via attention.

|                          | Word2Vec (Module 5)                   | BERT / DistilBERT                |
| ------------------------ | ------------------------------------- | -------------------------------- |
| Same word, two sentences | Same vector                           | Different vectors                |
| Training                 | Predict neighbours from co-occurrence | Masked language modelling        |
| Speed                    | Fast, small                           | Slower, larger                   |
| Context window           | Fixed (5–10 words)                    | Full sentence (up to 512 tokens) |

### Pre-training objectives

BERT is pre-trained on massive unlabelled text (Wikipedia + BookCorpus) with two objectives:

1. **Masked Language Modelling (MLM):** randomly mask 15% of input tokens and predict them from bidirectional context. This forces the model to understand words from both left and right context simultaneously - unlike earlier language models that read left-to-right only.

2. **Next Sentence Prediction (NSP):** given two sentences, predict whether sentence B actually follows sentence A. This teaches the model inter-sentence relationships (later work found NSP less important, but it was part of the original recipe).

After pre-training, BERT is **fine-tuned** on labelled task data by adding a task-specific head (classification, NER, QA) and training on labelled examples. This is **transfer learning** - language understanding learned from billions of unlabelled words transfers to tasks with limited labelled data.

### DistilBERT

Sanh et al. (2019) created **DistilBERT** via **knowledge distillation**: a smaller "student" model (6 layers) is trained to mimic the full "teacher" BERT (12 layers). The key insight is that the student is trained not on hard labels (correct/incorrect) but on the teacher's full probability distribution over all classes. These "soft targets" contain richer information — they reveal which wrong answers the teacher considered plausible. A student trained on soft targets learns more from each example than one trained on hard labels alone.

Result: ~40% smaller, ~60% faster, ~97% of BERT's performance on **GLUE** (General Language Understanding Evaluation) benchmarks. GLUE is a standard suite of 9 NLP tasks — including sentiment analysis, textual entailment, and paraphrase detection — used to benchmark language models. "97% of BERT's score" means DistilBERT matches the full model on nearly every task despite being 40% smaller. Good default for this course.

---

## Tokenization - splitting text the transformer way

NLTK splits on whitespace. Transformers use **subword tokenization** to handle open vocabularies - any word can be represented as a sequence of known subword pieces, even if the exact word was never seen during training.

| Tool              | Algorithm                | Used for                               |
| ----------------- | ------------------------ | -------------------------------------- |
| **tiktoken**      | BPE (Byte-Pair Encoding) | OpenAI models                          |
| **AutoTokenizer** | WordPiece                | Hugging Face models (BERT, DistilBERT) |

### Byte-Pair Encoding (BPE)

Sennrich, Haddow, and Birch (2016) adapted BPE for NLP. The algorithm:

1. Start with individual characters (or bytes) as the vocabulary
2. Count all adjacent symbol pairs in the training corpus
3. Merge the most frequent pair into a new symbol
4. Repeat until vocabulary reaches the desired size

**Worked example** on corpus `{low, lower, newest, widest}`:

| Step  | Merge              | Vocabulary grows                 |
| ----- | ------------------ | -------------------------------- |
| Start | -                  | `{l, o, w, e, r, n, s, t, i, d}` |
| 1     | `e` + `s` → `es`   | add `es`                         |
| 2     | `es` + `t` → `est` | add `est`                        |
| 3     | `l` + `o` → `lo`   | add `lo`                         |
| 4     | `lo` + `w` → `low` | add `low`                        |
| …     | …                  | …                                |

After training, `lowest` tokenises as `[low, est]` and `newer` as `[new, er]` - common subwords are shared across related words.

```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode("Reeves near the docks")
print(enc.decode(tokens))  # round-trip
print(len(tokens))         # token count for API limits
```

### WordPiece

WordPiece (used by BERT) is similar to BPE but merges are chosen to **maximise the likelihood** of the training corpus rather than by raw pair frequency. Continuation tokens are marked with `##` prefix: `tokenizing` → `[token, ##izing]`.

Different models produce different tokenisations for the same text. Exercise 02 compares tiktoken vs DistilBERT side by side.

---

## Hugging Face pipelines - deploy in minutes

The `pipeline` API wraps model, tokenizer, and post-processing into a single call:

```python
from transformers import pipeline

sentiment = pipeline("sentiment-analysis")
result = sentiment("I saw him near the docks on Tuesday.")
# [{'label': 'NEGATIVE', 'score': 0.82}, ...]

ner = pipeline("ner", grouped_entities=True)
ner("Margaret Hayes saw Reeves near the docks.")

zero_shot = pipeline("zero-shot-classification")
zero_shot("He looked nervous.", candidate_labels=["calm", "hostile"])
```

| Pipeline                   | Task                       | Module contrast             |
| -------------------------- | -------------------------- | --------------------------- |
| `sentiment-analysis`       | Positive/negative          | Module 3 trained classifier |
| `ner`                      | Entity spans               | Module 6 spaCy NER          |
| `zero-shot-classification` | Custom labels, no training | Module 3 requires labels    |

Under the hood, each pipeline loads a pre-trained model, tokenises input, runs inference, and formats output. You get transformer-quality results without writing training code.

---

## Zero-shot classification - no training data required

Provide **candidate labels** at inference time. The model scores how well each label describes the text using natural language inference - useful when you lack labelled Inkwell data:

```python
classifier = pipeline("zero-shot-classification")
classifier(
    "I do not appreciate being dragged in again.",
    candidate_labels=["calm", "hostile"],
)
```

Compare to Module 3's TF-IDF + Naive Bayes - zero-shot needs no `fit()`, but may miss domain nuance that a fine-tuned model would capture.

---

## Fine-tuning - training on the agency archives (optional)

Pre-trained models are generalists. **Fine-tuning** adapts them to your specific task by continuing training on labelled data with a task-specific head:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2
)
# ... prepare Dataset, Trainer, train ...
```

The workflow:

1. Load pre-trained DistilBERT (language understanding already learned)
2. Add a classification head (randomly initialised)
3. Train on labelled examples (`witness_sentiment.json` - calm vs hostile)
4. Evaluate against Module 3's TF-IDF baseline

Fine-tuning typically needs hundreds to thousands of labelled examples and benefits from GPU acceleration. On CPU it works but takes several minutes.

---

## Transfer learning - why pre-training works

The central idea behind BERT, GPT (Radford et al., 2018), and all modern NLP models:

1. **Pre-train** on massive unlabelled text to learn general language structure (grammar, semantics, world knowledge)
2. **Fine-tune** on small labelled datasets for specific tasks (sentiment, NER, classification)

Language understanding is **transferable** - a model that learns what "not guilty" means from reading Wikipedia can apply that understanding to witness statements with minimal additional training. This is why transformers with 110M+ parameters outperform classical methods trained from scratch on 50 labelled examples.

---

## Field rules

- **Start with a pipeline before you fine-tune.** Baseline quality first, then improve.
- **Fine-tuning needs labelled data and compute.** DistilBERT is a good starting point for speed.
- **Watch token limits.** DistilBERT max length is 512 tokens - long statements need truncation.
- **Model downloads are cached.** First run is slow; subsequent runs use `~/.cache/huggingface/`.
- **Exercise 03 is optional.** Requires `[local-ml]` and several minutes on CPU.

---

## Demo

Interactive console menu - pipelines, tokenizers, and zero-shot on Inkwell data:

```bash
python module-07-transformers/demo/demo.py
```

---

## Exercises

All exercises run transformer models locally and require `[local-ml]` (PyTorch + Hugging Face).

| Folder                                                    | Part A (Inkwell)                              | Part B (optional `--real-world`)       |
| --------------------------------------------------------- | --------------------------------------------- | -------------------------------------- |
| [`exercises/01-hf-pipelines`](exercises/01-hf-pipelines/) | Inference lab: sentiment, NER, zero-shot, summarisation | SMS spam + CoNLL NER   |
| [`exercises/02-tokenization`](exercises/02-tokenization/) | Text generation with distilgpt2               | Model comparison (distilgpt2 vs gpt2)  |
| [`exercises/03-fine-tuning`](exercises/03-fine-tuning/)   | Fine-tune DistilBERT on witness sentiment     | Movie reviews sample                   |

Exercises are **independent** - complete them in any order.

Run an exercise:

```bash
python module-07-transformers/exercises/01-hf-pipelines/start.py
python module-07-transformers/exercises/02-tokenization/start.py
python module-07-transformers/exercises/03-fine-tuning/start.py
python module-07-transformers/exercises/01-hf-pipelines/start.py --real-world
```

Run tests (from each exercise folder):

```bash
cd module-07-transformers/exercises/01-hf-pipelines && pytest test_start.py test_extension.py -v
cd module-07-transformers/exercises/02-tokenization && pytest test_start.py test_extension.py -v
cd module-07-transformers/exercises/03-fine-tuning && pytest test_start.py test_extension.py -v
```

## Slides

From repo root: `pnpm slides:07`, or `cd module-07-transformers/slides && pnpm dev`.

## Reference

- [Hugging Face - Transformers documentation](https://huggingface.co/docs/transformers/index)
- [Hugging Face - Pipelines](https://huggingface.co/docs/transformers/pipeline_tutorial)
- [Hugging Face - Fine-tuning](https://huggingface.co/docs/transformers/training)
- [OpenAI - tiktoken](https://github.com/openai/tiktoken)
- Vaswani, A., et al. (2017). Attention is all you need. _NeurIPS_, 5998–6008.
- Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. _NAACL_, 4171–4186.
- Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT. _NeurIPS Workshop_.
- Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units. _ACL_, 1715–1725.
- Radford, A., et al. (2018). Improving language understanding by generative pre-training. _OpenAI Technical Report_.
