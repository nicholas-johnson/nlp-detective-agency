# Module 1 — Text Preprocessing and Cleaning

> The case files are a mess — coffee stains, shorthand, inconsistent formatting, and enough typos to make a proofreader weep. Before any analysis can begin, every document needs to be cleaned, normalised, and tokenised into a consistent form. This module is your evidence prep room at Inkwell Investigations: turn chaotic prose into machine-readable text.

## Learning goals

- **Tokenise** text at the word and sentence level using NLTK and spaCy.
- Apply **case folding**, **normalisation**, and **regex cleaning** to raw documents.
- Remove or handle **stopwords**, **punctuation**, **numbers**, and **unicode** artefacts.
- Compare **stemming** vs **lemmatisation** and know when to use each.
- Build a reusable **preprocessing pipeline** that chains these steps together.

## Setup

Install NLP dependencies and download NLTK data (one-time):

```bash
pip install -e ".[nlp,dev]"
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

---

## Tokenisation — breaking text into pieces

A witness statement is a wall of prose. Tokenisation splits it into sentences and words so you can examine each piece individually.

**Sentence tokenisation** finds sentence boundaries. Punctuation is trickier than it looks — "Dr. Reeves arrived at 4 p.m." is one sentence, not four.

```python
from nltk.tokenize import sent_tokenize

text = "I saw HIM near the DOCKS on Tuesday!!! He was with a man called Reeves."
sentences = sent_tokenize(text)
# ['I saw HIM near the DOCKS on Tuesday!!!', 'He was with a man called Reeves.']
```

**Word tokenisation** splits text into tokens. NLTK's `word_tokenize` handles contractions and punctuation as separate tokens:

```python
from nltk.tokenize import word_tokenize

tokens = word_tokenize("Reeves—or so I thought.")
# ['Reeves', '—', 'or', 'so', 'I', 'thought', '.']
```

**NLTK vs spaCy** — both tokenise, but they serve different workflows:

| | NLTK | spaCy |
| --- | --- | --- |
| Best for | Learning, quick scripts, classical NLP | Production pipelines, NER, POS tagging |
| Setup | `nltk.download('punkt_tab')` | `python -m spacy download en_core_web_sm` |
| Output | Lists of strings | `Doc` object with rich annotations |

```python
# spaCy comparison (optional — requires model download)
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("I saw him near the docks on Tuesday.")
tokens = [t.text for t in doc]
```

For this module's exercises we use NLTK. You will meet spaCy again in Module 6.

---

## Cleaning and normalisation — scrubbing the grime

Real case files arrive messy. Cleaning transforms them into a consistent baseline before tokenisation.

**Case folding** — convert to lowercase so "DOCKS" and "docks" match:

```python
text = text.lower()
```

**Regex cleaning** — remove redactions, collapse whitespace, strip case references:

```python
import re

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[redacted\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"case-\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

**Unicode normalisation** — curly quotes, em-dashes, and accented characters can split what should be the same token:

```python
import unicodedata

text = unicodedata.normalize("NFKC", text)
```

Apply cleaning **before** tokenisation. Tokenise the normalised text, not the raw archive copy — but always keep the original for audit trails.

---

## Stopwords, stemming, and lemmatisation — trimming the noise

"the", "a", "is" tell you nothing about a suspect. Stopwords and word-form reduction strip noise so distinctive terms stand out.

**Stopwords** — high-frequency words with little semantic value:

```python
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))
tokens = [t for t in tokens if t.isalpha() and t not in stops]
```

**Stemming** — chop suffixes with rules. Fast but crude:

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
[stemmer.stem(w) for w in ["investigating", "investigated", "investigation"]]
# ['investig', 'investig', 'investig']
```

**Lemmatisation** — reduce to dictionary forms using vocabulary and POS. Slower but more accurate:

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("investigating", pos="v")  # 'investigate'
lemmatizer.lemmatize("docks")                    # 'dock'
```

Use `pos='v'` for verbs, `pos='a'` for adjectives. Default is noun.

**When to use which:**

| Technique | Speed | Accuracy | Good for |
| --- | --- | --- | --- |
| Stemming | Fast | Lower | Search indexing, bag-of-words |
| Lemmatisation | Slower | Higher | Topic modelling, briefings, reporting |

---

## Building a preprocessing pipeline — the assembly line

One-off cleaning scripts do not scale. Chain steps into a reusable function:

```python
def preprocess_statement(text: str) -> list[str]:
    text = normalize_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = remove_stopwords(tokens)
    return lemmatize_tokens(tokens)
```

Each step is a pure function — input text in, transformed output out. That makes testing easy: you can verify `normalize_text` independently of `lemmatize_tokens`.

For batch work, map the pipeline across a list of documents and aggregate results (word frequencies, audit counts) with `collections.Counter`.

---

## Field rules

- **Never preprocess your test set with statistics from training data.** Fit on train, transform on test.
- **Stemming is fast but crude; lemmatisation is slower but accurate.** Pick based on your downstream task.
- **Keep the original text.** Always store raw text alongside cleaned versions for audit trails.

---

## Demo

Interactive console menu — step through each technique on Inkwell witness statements:

```bash
python module-01-text-preprocessing/demo/demo.py
```

---

## Exercises

| Folder | Mission |
| ------ | ------- |
| [`exercises/01-statement-audit`](exercises/01-statement-audit/) | Batch-audit every witness statement: sentence counts, word counts, flag statements that need review. |
| [`exercises/02-case-briefing`](exercises/02-case-briefing/) | Build a preprocessing pipeline and generate a word-frequency briefing for a single case. |

Run an exercise interactively:

```bash
python module-01-text-preprocessing/exercises/01-statement-audit/start.py
python module-01-text-preprocessing/exercises/02-case-briefing/start.py CASE-42
```

Run tests for this module (from each exercise folder):

```bash
cd module-01-text-preprocessing/exercises/01-statement-audit && pytest test_start.py -v
cd module-01-text-preprocessing/exercises/02-case-briefing && pytest test_start.py -v
```

## Slides

From repo root: `pnpm slides:01`, or `cd module-01-text-preprocessing/slides && pnpm dev`.

## Reference

- [NLTK — Tokenizing text](https://www.nltk.org/book/ch03.html)
- [spaCy — Linguistic Features](https://spacy.io/usage/linguistic-features)
- [NLTK — Stemming and Lemmatization](https://www.nltk.org/book/ch05.html)
