# Module 1 - Text Preprocessing and Cleaning

> The case files are a mess - coffee stains, shorthand, inconsistent formatting, and enough typos to make a proofreader weep. Before any analysis can begin, every document needs to be cleaned, normalised, and tokenised into a consistent form. This module is your evidence prep room at Inkwell Investigations: turn chaotic prose into machine-readable text.

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

## Why preprocess? - the tradeoff

Raw text is not ready for counting, comparing, or classifying. Every downstream technique in this course - bag-of-words vectors (Module 2), classifiers (Module 3), topic models (Module 4) - assumes that input has been reduced to a consistent set of tokens.

Preprocessing always involves a tradeoff:

| You gain                                               | You lose                                                   |
| ------------------------------------------------------ | ---------------------------------------------------------- |
| Fewer spurious matches (`DOCKS` = `docks`)             | Original casing and punctuation context                    |
| Smaller vocabulary (stopword removal)                  | Function words that sometimes carry sentiment ("not good") |
| Related forms grouped (`investigate`, `investigating`) | Distinctions between word senses (`bank` river vs money)   |

**Zipf's law** explains why stopword removal helps. In almost every natural language corpus, word frequency follows a power law: a few words (`the`, `a`, `is`) account for a huge share of all tokens, while the long tail contains thousands of rare words that appear once or twice. If you count raw tokens without filtering, your features are dominated by function words that carry almost no investigative signal. Removing stopwords shifts attention to the distinctive terms - `docks`, `ledger`, `Reeves` - that actually differentiate documents.

Apply preprocessing **deliberately**, not automatically. Keep the original text for audit trails. Fit any statistics (vocabulary, stopword lists) on training data only.

---

## Tokenisation - breaking text into pieces

A witness statement is a wall of prose. Tokenisation splits it into sentences and words so you can examine each piece individually. This is the first irreversible decision in most NLP pipelines: once you tokenise, you cannot recover the original whitespace or punctuation layout from the token list alone.

### Sentence tokenisation

**Sentence tokenisation** finds sentence boundaries. Punctuation is trickier than it looks - "Dr. Reeves arrived at 4 p.m." is one sentence, not four. Abbreviations, decimal numbers, ellipses, and quoted speech all create ambiguous periods.

NLTK's `sent_tokenize` uses the **Punkt** algorithm (Kiss & Strunk, 2006), a statistical model trained on the Penn Treebank Wall Street Journal corpus. Punkt learns where sentence boundaries likely occur based on local context - not just "period = end of sentence". That is why it handles `Dr.` and `4 p.m.` correctly without a hand-maintained abbreviation list.

```python
from nltk.tokenize import sent_tokenize

text = "I saw HIM near the DOCKS on Tuesday!!! He was with a man called Reeves."
sentences = sent_tokenize(text)
# ['I saw HIM near the DOCKS on Tuesday!!!', 'He was with a man called Reeves.']
```

### Word tokenisation

**Word tokenisation** splits text into tokens. NLTK's `word_tokenize` follows **Penn Treebank conventions**: contractions become two tokens (`"don't"` → `do`, `'`, `n't`), punctuation is separated, and hyphenated compounds may split depending on context.

```python
from nltk.tokenize import word_tokenize

tokens = word_tokenize("Reeves-or so I thought.")
# ['Reeves', '-', 'or', 'so', 'I', 'thought', '.']
```

Different tokenisers produce different token counts for the same text. That matters when you compare results across tools or reproduce published benchmarks - always note which tokeniser you used.

### NLTK vs spaCy

|              | NLTK                                   | spaCy                                     |
| ------------ | -------------------------------------- | ----------------------------------------- |
| Best for     | Learning, quick scripts, classical NLP | Production pipelines, NER, POS tagging    |
| Setup        | `nltk.download('punkt_tab')`           | `python -m spacy download en_core_web_sm` |
| Output       | Lists of strings                       | `Doc` object with rich annotations        |
| Tokenisation | Rule + statistical (Punkt)             | Data-driven, integrated with parser       |

```python
# spaCy comparison (optional - requires model download)
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("I saw him near the docks on Tuesday.")
tokens = [t.text for t in doc]
```

For this module's exercises we use NLTK. You will meet spaCy again in Module 6.

---

## Cleaning and normalisation - scrubbing the grime

Real case files arrive messy. Cleaning transforms them into a consistent baseline before tokenisation. The goal is not to make text "pretty" - it is to ensure that tokens that should match _do_ match, and that artefacts from OCR, copy-paste, or formatting do not create spurious vocabulary entries.

### Case folding

**Case folding** converts text to lowercase so `DOCKS` and `docks` become the same token. This is lossy: proper nouns (`Reeves`) lose their capitalisation signal, which NER systems (Module 6) rely on. For bag-of-words pipelines, the tradeoff usually favours folding.

```python
text = text.lower()
```

### Regex cleaning

**Regex cleaning** removes known noise patterns - redactions, case ID stamps, excess whitespace - before tokenisation:

```python
import re

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\[redacted\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"case-\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

Design regex rules for _your_ data. Inkwell statements contain `[REDACTED]` blocks and `CASE-42` references; movie reviews might contain HTML entities or star-rating markers. Inspect raw text before writing rules.

### Unicode normalisation

Computers can represent the same visual character in multiple byte sequences. The name `José` might be stored as a single pre-composed code point (U+00E9, é) or as `e` + combining acute accent (U+0065 U+0301). Without normalisation, these tokenise differently and never match in a vocabulary.

Unicode defines several normalisation forms:

| Form     | What it does                                                             | When to use                                         |
| -------- | ------------------------------------------------------------------------ | --------------------------------------------------- |
| **NFC**  | Composed form - merge base + combining marks                             | General text; preserves meaning                     |
| **NFKC** | Compatibility composed - also folds ligatures (ﬁ → fi), fullwidth digits | Noisy web/OCR text; aggressive matching             |
| **NFD**  | Decomposed form                                                          | Rare; used when you need to inspect combining marks |

```python
import unicodedata

text = unicodedata.normalize("NFKC", text)
```

Use **NFKC** when you want maximum token overlap across sources with inconsistent encoding. Use **NFC** when you need to preserve characters that NFKC would collapse. For witness names and international text, inspect a sample before choosing.

Apply cleaning **before** tokenisation. Tokenise the normalised text, not the raw archive copy - but always keep the original for audit trails.

---

## Stopwords, stemming, and lemmatisation - trimming the noise

"the", "a", "is" tell you nothing about a suspect. Stopwords and word-form reduction strip noise so distinctive terms stand out. Each technique removes a different kind of redundancy.

### Stopwords

**Stopwords** are high-frequency function words with little semantic content for most classification and topic tasks. NLTK ships a list of 179 English stopwords derived from the Brown corpus.

```python
from nltk.corpus import stopwords

stops = set(stopwords.words("english"))
tokens = [t for t in tokens if t.isalpha() and t not in stops]
```

Be cautious with negation: removing `not` turns "not guilty" into "guilty". For sentiment tasks (Module 3), consider keeping negation words or using bigrams (`not guilty` as a single feature in Module 2).

### Stemming - Porter's algorithm

**Stemming** strips suffixes with a cascade of rules. Porter (1980) defined a five-phase English stemmer that maps inflected forms to a common stem:

| Phase | Rule type           | Example                                            |
| ----- | ------------------- | -------------------------------------------------- |
| 1a    | Plural / past tense | `investigations` → `investigation`                 |
| 1b    | `-ed`, `-ing`       | `investigation` → `investig` (after further rules) |
| 2–4   | Suffix stripping    | `-ational` → `-ate`, `-izer` → `-ize`              |
| 5     | Final `-e`, `-ll`   | `investigate` → `investig`                         |

Worked example on **"investigations"**:

1. Phase 1a: `investigations` → `investigation` (remove plural `-s`)
2. Phase 2: `investigation` → `investigate` (strip `-ion` → `-ate` pattern)
3. Phase 5: `investigate` → `investig` (remove final `-e`)

```python
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
[stemmer.stem(w) for w in ["investigating", "investigated", "investigation"]]
# ['investig', 'investig', 'investig']
```

All three forms collapse to `investig`. That is the goal for search indexing - but stems are not always valid English words (`studies` → `studi`).

The **Snowball stemmer** is Porter's successor: it supports multiple languages and refines some English rules. NLTK's `PorterStemmer` implements the original 1980 algorithm; `SnowballStemmer("english")` is the improved variant.

### Lemmatisation - dictionary lookup

**Lemmatisation** reduces words to their dictionary form (lemma) using vocabulary and part-of-speech information. Miller (1995) created **WordNet**, a lexical database where words are organised into **synsets** - sets of synonyms sharing a meaning. The lemmatiser looks up each token in WordNet and returns the canonical form for that POS tag.

Why POS matters: `saw` as a verb lemmatises to `see`; `saw` as a noun stays `saw`. Without POS, the lemmatiser defaults to noun and produces wrong results.

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("investigating", pos="v")  # 'investigate'
lemmatizer.lemmatize("docks")                    # 'dock'
lemmatizer.lemmatize("saw", pos="v")             # 'see'
lemmatizer.lemmatize("saw", pos="n")             # 'saw'
```

Use `pos='v'` for verbs, `pos='a'` for adjectives, `pos='r'` for adverbs. Default is noun.

NLTK uses WordNet's **Morphy** function internally - a rule-based morphological analyser that tries suffix stripping patterns and validates results against the WordNet lexicon.

### When to use which

| Technique     | Speed                                  | Accuracy                                 | Good for                              |
| ------------- | -------------------------------------- | ---------------------------------------- | ------------------------------------- |
| Stemming      | Fast                                   | Lower - stems may be non-words           | Search indexing, bag-of-words         |
| Lemmatisation | Slower - requires POS + lexicon lookup | Higher - produces valid dictionary forms | Topic modelling, briefings, reporting |

---

## Building a preprocessing pipeline - the assembly line

One-off cleaning scripts do not scale. Chain steps into a reusable function where each stage is a pure function - input text in, transformed output out:

```python
def preprocess_statement(text: str) -> list[str]:
    text = normalize_text(text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = remove_stopwords(tokens)
    return lemmatize_tokens(tokens)
```

This design makes testing easy: verify `normalize_text` independently of `lemmatize_tokens`. It also mirrors the scikit-learn `Pipeline` pattern you will use in Module 3 - fit on training data, transform on test data, never leak statistics across the split.

For batch work, map the pipeline across a list of documents and aggregate results (word frequencies, audit counts) with `collections.Counter`.

---

## Field rules

- **Never preprocess your test set with statistics from training data.** Fit on train, transform on test.
- **Stemming is fast but crude; lemmatisation is slower but accurate.** Pick based on your downstream task.
- **Keep the original text.** Always store raw text alongside cleaned versions for audit trails.

---

## Demo

Interactive console menu - step through each technique on Inkwell witness statements:

```bash
python module-01-text-preprocessing/demo/demo.py
```

---

## Exercises

| Folder                                                          | Mission                                                                                              |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| [`exercises/01-statement-audit`](exercises/01-statement-audit/) | Batch-audit every witness statement: sentence counts, word counts, flag statements that need review. |
| [`exercises/02-case-briefing`](exercises/02-case-briefing/)     | Build a preprocessing pipeline and generate a word-frequency briefing for a single case.             |
| [`exercises/03-review-scanner`](exercises/03-review-scanner/)   | Apply preprocessing to real movie reviews and compare positive vs negative vocabulary.               |

Run an exercise interactively:

```bash
python module-01-text-preprocessing/exercises/01-statement-audit/start.py
python module-01-text-preprocessing/exercises/02-case-briefing/start.py CASE-42
python module-01-text-preprocessing/exercises/03-review-scanner/start.py
```

Run tests for this module (from each exercise folder):

```bash
cd module-01-text-preprocessing/exercises/01-statement-audit && pytest test_start.py -v
cd module-01-text-preprocessing/exercises/02-case-briefing && pytest test_start.py -v
cd module-01-text-preprocessing/exercises/03-review-scanner && pytest test_start.py -v
```

## Slides

From repo root: `pnpm slides:01`, or `cd module-01-text-preprocessing/slides && pnpm dev`.

## Reference

- [NLTK - Tokenizing text](https://www.nltk.org/book/ch03.html)
- [spaCy - Linguistic Features](https://spacy.io/usage/linguistic-features)
- [NLTK - Stemming and Lemmatization](https://www.nltk.org/book/ch05.html)
- Porter, M. F. (1980). An algorithm for suffix stripping. _Program_, 14(3), 130–137.
- Miller, G. A. (1995). WordNet: A lexical database for English. _Communications of the ACM_, 38(11), 39–41.
- Kiss, T., & Strunk, J. (2006). Unsupervised multilingual sentence boundary detection. _Computational Linguistics_, 32(4), 485–525.
