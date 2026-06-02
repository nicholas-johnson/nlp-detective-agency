# Python NLP: Inkwell Investigations

```
   ___       _               _ _
  |_ _|_ __ | | ____      __| | |
   | || '_ \| |/ /\ \ /\ / / | |
   | || | | |   <  \ V  V /|_| |
  |___|_| |_|_|\_\  \_/\_/ (_)_|
  Investigations — NLP Division
```

**Mission:** Build text analysis systems that turn chaotic prose into solved cases — using classical NLP, machine learning, and modern transformers, all in **Python**. You have just joined Inkwell Investigations as a data analyst. The agency is drowning in decades of unsorted witness statements, tip-off letters, and cold-case files. Your job: build the tools that bring order to the chaos.

## Prerequisites

- **Python** 3.12 or newer — check with `python3 --version` or `py --version`
- **pip** (bundled with Python — used inside a virtual environment)
- **Node.js** v20+ and **pnpm** v10+ (for slides only) — `corepack enable && corepack prepare pnpm@latest --activate`
- A code editor (VS Code / Cursor recommended)

## Setup

### Python environment

Create a virtual environment and install dependencies:

```bash
cd nlp-python-course

# Create a virtual environment (use whichever python command gives you 3.12+)
python3 -m venv .venv        # or: py -m venv .venv

# Activate it
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate     # Windows

# Install NLP dependencies + dev tools
pip install -e ".[nlp,dev]"

# Download NLTK data (one-time)
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"

# Download spaCy model (needed from Module 6 onwards)
python -m spacy download en_core_web_sm
```

**Optional extras** (pick what you need):

```bash
pip install -e ".[nlp,local-ml,dev]"    # adds PyTorch, HF transformers (Modules 7–8)
```

> **Tip:** If `python3 --version` shows an old version but `py --version` shows 3.12+, use `py` instead. On macOS with Homebrew Python you **must** use a virtual environment — pip will refuse to install system-wide.

You need to activate the venv (`source .venv/bin/activate`) each time you open a new terminal.

### Slides dependencies (pnpm monorepo)

```bash
pnpm install
```

### Running tests

```bash
# Run all exercise tests (many fail until you complete start.py)
pytest

# Run tests for a single module
pytest module-01-text-preprocessing/

# Run a single exercise's tests
cd module-03-text-classification/exercises/01-sentiment-triage && pytest test_start.py -v
```

## Project structure

This is a **hybrid monorepo** — Python exercises and demos live alongside a pnpm workspace that powers the slide decks.

Each **module** has its own `README.md` (full tutorial), **demo** scripts you can run with `python …`, and **exercises** with `start.py` (your work), `test_start.py` (pytest), and `solution.py` (instructor reference — try the exercise first!).

Shared data lives in [`data/`](data/) — Inkwell case files in `data/inkwell/` and real public datasets in `data/public/`.

## Slides

Each module includes a Vite app under `slides/` that renders teaching decks with the workspace package [`slide-deck`](slide-deck/).

```bash
pnpm slides:00          # same pattern :00 … :07, :capstone
# or
cd module-04-topic-modelling/slides && pnpm dev
```

## Schedule

### Day 1 — Classical NLP foundations

| Block | Module | Topic |
| ----- | ------ | ----- |
| 0 | [module-00-python-for-text](module-00-python-for-text/) | Strings, regex, file I/O, Counter — the analyst's toolkit |
| 1 | [module-01-text-preprocessing](module-01-text-preprocessing/) | Tokenisation, cleaning, stemming, lemmatisation |
| 2 | [module-02-feature-extraction](module-02-feature-extraction/) | Bag-of-Words, n-grams, TF-IDF, cosine similarity |
| 3 | [module-03-text-classification](module-03-text-classification/) | Naive Bayes, Logistic Regression, SVM, evaluation metrics |
| 4 | [module-04-topic-modelling](module-04-topic-modelling/) | LDA, NMF, perplexity, topic interpretation — the Day 1 closer |

### Day 2 — Meaning, structure, and modern models

| Block | Module | Topic |
| ----- | ------ | ----- |
| 5 | [module-05-word-embeddings](module-05-word-embeddings/) | Word2Vec, GloVe, similarity, analogies, document vectors |
| 6 | [module-06-ner-and-pos](module-06-ner-and-pos/) | spaCy POS tagging, dependency parsing, NER, EntityRuler |
| 7 | [module-07-transformers](module-07-transformers/) | Attention, BERT, HF pipelines, tokenizers, optional fine-tuning |
| 8 | [module-08-capstone](module-08-capstone/) | End-to-end pipeline: choose data, train, compare, deploy FastAPI |

## Course outline

All **exercises** run in **Python** and are checked with **pytest** (`start.py` / `test_start.py`). **Demos** are interactive `python …` scripts. Optional **slides** under each module's `slides/` folder are separate Vite + React apps for teaching only.

### Module 0 — [Python for Text Analysis](module-00-python-for-text/)

**Topics:** Strings (slicing, methods, f-strings), regular expressions (`re`), file I/O (`pathlib`, JSON, encoding), collections (`Counter`, `defaultdict`), comprehensions, type hints, CLI basics (`sys.argv`, `argparse`), NLP library survey.

| Exercise | Folder | What you build |
| -------- | ------ | -------------- |
| String sleuthing | [`exercises/01-string-sleuthing`](module-00-python-for-text/exercises/01-string-sleuthing/) | Slice, search, normalise witness strings |
| Regex extraction | [`exercises/02-regex-extraction`](module-00-python-for-text/exercises/02-regex-extraction/) | Extract dates, case IDs, redacted markers |
| Corpus loader | [`exercises/03-corpus-loader`](module-00-python-for-text/exercises/03-corpus-loader/) | Load JSON data, word frequencies with Counter |

### Module 1 — [Text Preprocessing](module-01-text-preprocessing/)

**Topics:** Tokenisation (NLTK Punkt, spaCy), case folding, regex cleaning, Unicode normalisation, stopwords, stemming (Porter), lemmatisation (WordNet), preprocessing pipelines.

| Exercise | Folder | What you build |
| -------- | ------ | -------------- |
| Statement audit | [`exercises/01-statement-audit`](module-01-text-preprocessing/exercises/01-statement-audit/) | Batch-audit witness statements: counts, flags |
| Case briefing | [`exercises/02-case-briefing`](module-01-text-preprocessing/exercises/02-case-briefing/) | Preprocessing pipeline + word-frequency briefing |
| Review scanner | [`exercises/03-review-scanner`](module-01-text-preprocessing/exercises/03-review-scanner/) | Preprocess real movie reviews, compare vocabularies |

### Module 2 — [Feature Extraction](module-02-feature-extraction/)

**Topics:** Vector space model, Bag-of-Words, n-grams, TF-IDF (derivation, variants, smoothing), cosine similarity, sparsity, scikit-learn vectorisers.

| Exercise | Folder | What you build |
| -------- | ------ | -------------- |
| Document fingerprints | [`exercises/01-document-fingerprints`](module-02-feature-extraction/exercises/01-document-fingerprints/) | BoW fingerprint card per statement |
| Matching prints | [`exercises/02-matching-prints`](module-02-feature-extraction/exercises/02-matching-prints/) | TF-IDF + cosine to find similar witnesses |
| Article matcher | [`exercises/03-article-matcher`](module-02-feature-extraction/exercises/03-article-matcher/) | TF-IDF similarity on real newsgroup articles |

### Module 3 — [Text Classification](module-03-text-classification/)

**Topics:** Train/test splits, sklearn pipelines, Naive Bayes (Bayes' theorem), Logistic Regression (sigmoid), SVM (margin), precision/recall/F1, confusion matrices, cross-validation.

| Exercise | Folder | What you build |
| -------- | ------ | -------------- |
| Sentiment triage | [`exercises/01-sentiment-triage`](module-03-text-classification/exercises/01-sentiment-triage/) | Classify calm vs hostile witness statements |
| Hoax filter | [`exercises/02-hoax-filter`](module-03-text-classification/exercises/02-hoax-filter/) | Compare classifiers on tip credibility |
| Spam detector | [`exercises/03-spam-detector`](module-03-text-classification/exercises/03-spam-detector/) | Classifier shootout on real SMS spam |

### Module 4 — [Topic Modelling](module-04-topic-modelling/)

**Topics:** Unsupervised discovery, LDA (generative story, Dirichlet), NMF (V=WH), perplexity, topic coherence, pyLDAvis, interpreting and labelling topics.

| Exercise | Folder | What you build |
| -------- | ------ | -------------- |
| Archive themes | [`exercises/01-archive-themes`](module-04-topic-modelling/exercises/01-archive-themes/) | LDA on cold cases, map files to themes |
| Evidence board | [`exercises/02-evidence-board`](module-04-topic-modelling/exercises/02-evidence-board/) | Compare LDA vs NMF, pick topic count |
| Real-world topics | [`exercises/03-real-world-topics`](module-04-topic-modelling/exercises/03-real-world-topics/) | Topic model 20 Newsgroups, audit vs real categories |

### Module 5 — [Word Embeddings](module-05-word-embeddings/)

**Topics:** Distributional semantics (Firth, Harris), Word2Vec (CBOW, Skip-gram, negative sampling), GloVe (co-occurrence), document vectors, static vs contextual embeddings, optional API embeddings.

| Exercise | Folder | What you build |
| -------- | ------ | -------------- |
| Alias map | [`exercises/01-alias-map`](module-05-word-embeddings/exercises/01-alias-map/) | Train Word2Vec, explore alias clusters |
| Embedding compass | [`exercises/02-embedding-compass`](module-05-word-embeddings/exercises/02-embedding-compass/) | GloVe similarity, analogies, odd-one-out |
| Semantic search | [`exercises/03-semantic-search`](module-05-word-embeddings/exercises/03-semantic-search/) | OpenAI embeddings — find pairs TF-IDF misses (optional) |

### Module 6 — [Linguistic Structure: POS and NER](module-06-ner-and-pos/)

**Topics:** spaCy pipeline, UPOS tagging, dependency parsing (SVO extraction), NER (BIO scheme, span-level evaluation), EntityRuler for custom entities.

| Exercise | Folder | What you build |
| -------- | ------ | -------------- |
| Grammar audit | [`exercises/01-grammar-audit`](module-06-ner-and-pos/exercises/01-grammar-audit/) | POS + SVO triples on statements |
| NER extraction | [`exercises/02-ner-extraction`](module-06-ner-and-pos/exercises/02-ner-extraction/) | Evidence board + gold-label P/R |
| Custom entities | [`exercises/03-custom-entities`](module-06-ner-and-pos/exercises/03-custom-entities/) | EntityRuler for case/ticket IDs |

### Module 7 — [Transformers and Pre-trained Models](module-07-transformers/)

**Topics:** Self-attention (Q/K/V), multi-head attention, positional encodings, BERT (MLM, NSP), DistilBERT, BPE/WordPiece tokenization, HF pipelines, zero-shot classification, optional fine-tuning, transfer learning.

| Exercise | Folder | What you build |
| -------- | ------ | -------------- |
| Inference lab | [`exercises/01-hf-pipelines`](module-07-transformers/exercises/01-hf-pipelines/) | Run sentiment, NER, zero-shot, summarisation locally |
| Text generation | [`exercises/02-tokenization`](module-07-transformers/exercises/02-tokenization/) | Load distilgpt2, generate continuations, explore temperature |
| Fine-tuning | [`exercises/03-fine-tuning`](module-07-transformers/exercises/03-fine-tuning/) | Fine-tune DistilBERT on witness sentiment |

### Module 8 — [Applied NLP Capstone](module-08-capstone/)

**Topics:** End-to-end classification pipeline, dataset choice (real public corpora), baseline vs transformer comparison, model persistence (joblib), FastAPI deployment, error analysis.

| Exercise | Folder | What you build |
| -------- | ------ | -------------- |
| Open case | [`exercises/01-open-case`](module-08-capstone/exercises/01-open-case/) | Choose data, train, compare, deploy as FastAPI service |

## License

Copyright (c) 2026 Nicholas Johnson. **All rights reserved.** This material is not licensed for use, copying, or distribution by others. See [LICENSE](LICENSE).
