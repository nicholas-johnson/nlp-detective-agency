# Exercise 01 - Statement Audit

The chief wants a quality report on every witness statement before analysis begins. Your job: load the Inkwell archive, clean and tokenise each statement, and flag anything that looks too long or too complex for a quick read-through.

This exercise focuses on **loading**, **cleaning**, **tokenising**, and **auditing** - not stopwords or lemmatisation (that comes in Exercise 02).

## Before you start

Make sure NLTK is installed and its data packages are downloaded (one-time setup):

```bash
pip install -e ".[nlp,dev]"
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

Open `start.py` in this folder. Each function has a `# TODO` - implement them in order.

## The data

Witness statements live in `data/inkwell/statements.json`. Each record looks like:

```json
{
  "id": "STM-001",
  "case_id": "CASE-42",
  "witness": "Margaret Hayes",
  "recorded": "1947-03-12",
  "raw_text": "I saw HIM near the DOCKS on Tuesday!!! ..."
}
```

The text is deliberately messy: ALL CAPS, extra spaces, `[REDACTED]` blocks, and `CASE-42` references scattered through the prose.

## What you'll build

When your functions are complete, running `start.py` prints an audit table like:

```
Inkwell Investigations - Statement Audit
========================================================================
ID         Witness                Sents Words  Review
------------------------------------------------------------------------
STM-001    Margaret Hayes             4    29
STM-009    Frank Holloway            17   131  YES
...
------------------------------------------------------------------------
10 statements audited, 2 flagged for review.
```

The `main()` function is already written - you only implement the helper functions it calls.

## Functions to implement

Work through these in order. Each builds on the previous one.

### 1. `load_statements(path)`

Read the JSON file and return the list of statement dicts.

```python
def load_statements(path: Path) -> list[dict]:
    return json.loads(path.read_text())
```

### 2. `normalize_text(text)`

Clean raw statement text before tokenisation:

1. Convert to **lowercase**
2. Remove `[REDACTED]` (case-insensitive)
3. Remove case references like `CASE-42`, `CASE-17` (pattern: `case-` followed by digits)
4. Collapse multiple whitespace characters into a single space and strip ends

Apply the steps in that order.

```python
text = text.lower()
text = re.sub(r"\[redacted\]", " ", text, flags=re.IGNORECASE)
text = re.sub(r"case-\d+", " ", text, flags=re.IGNORECASE)
text = re.sub(r"\s+", " ", text).strip()
```

### 3. `tokenize_sentences(text)` and `tokenize_words(text)`

Both functions should **normalise the text first**, then tokenise:

```python
from nltk.tokenize import sent_tokenize, word_tokenize

sentences = sent_tokenize(normalize_text(text))
tokens = word_tokenize(normalize_text(text))
```

- `sent_tokenize` splits on sentence boundaries (`.`, `!`, `?`)
- `word_tokenize` splits into words and punctuation as separate tokens

### 4. `audit_statement(statement)`

Given one statement dict, return an audit record:

```python
{
    "id": statement["id"],
    "witness": statement["witness"],
    "sentence_count": ...,   # len of tokenize_sentences(statement["raw_text"])
    "word_count": ...,       # count of alphabetic tokens only
    "needs_review": ...,     # True or False - see rules below
}
```

**Review rules** - flag a statement when **either**:

- `sentence_count > 4` (strictly greater than four), or
- `word_count > 120` (strictly greater than 120)

For `word_count`, count only tokens where `token.isalpha()` is `True`. Punctuation and numbers do not count.

```python
word_count = sum(1 for t in tokenize_words(raw) if t.isalpha())
needs_review = sentence_count > 4 or word_count > 120
```

A statement with exactly 4 sentences and 120 words should **not** be flagged.

### 5. `audit_archive(statements)`

Audit every statement in the list and return results **sorted by `id`**:

```python
audits = [audit_statement(s) for s in statements]
return sorted(audits, key=lambda a: a["id"])
```

---

## Run it

From the repo root:

```bash
python module-01-text-preprocessing/exercises/01-statement-audit/start.py
```

Or from this folder:

```bash
cd module-01-text-preprocessing/exercises/01-statement-audit
python start.py
```

## Run the tests

Tests import from `start.py`, so run pytest **from this exercise folder**:

```bash
cd module-01-text-preprocessing/exercises/01-statement-audit
pytest test_start.py -v
```

All 12 tests should pass when you are done.

---

## Optional hints

<details>
<summary>Hint: counting alphabetic words only</summary>

`word_tokenize` returns punctuation as separate tokens (`"Hello, world!"` → `["Hello", ",", "world", "!"]`). Filter with a comprehension:

```python
alpha_tokens = [t for t in tokens if t.isalpha()]
word_count = len(alpha_tokens)
# or in one line:
word_count = sum(1 for t in tokens if t.isalpha())
```

</details>

<details>
<summary>Hint: the review threshold is strict</summary>

`sentence_count > 4` means 4 sentences is fine, 5 is flagged. Same for words: 120 is fine, 121 is flagged. Statement `STM-009` (Frank Holloway) is long on both measures and should be flagged.

</details>

<details>
<summary>Hint: normalise before tokenising</summary>

Always pass text through `normalize_text` before `sent_tokenize` or `word_tokenize`. If you tokenise raw text, `DOCKS` and `docks` count as different tokens and your counts will be wrong.

</details>

<details>
<summary>Hint: regex flags</summary>

Use `flags=re.IGNORECASE` on the `[redacted]` pattern so `[REDACTED]`, `[redacted]`, and `[Redacted]` all match. The `case-\d+` pattern should also be case-insensitive since you lowercased first - but applying `IGNORECASE` does not hurt.

</details>

<details>
<summary>Hint: if sentence tokenisation looks wrong</summary>

NLTK needs the `punkt_tab` data package. If you see a `LookupError`, run:

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

Single-letter "sentences" like `"A. B. C. D."` are **not** split into four sentences - that is expected NLTK behaviour. Real witness prose works fine.

</details>

---

## Checklist

- [ ] `load_statements` returns a list of dicts from the JSON file
- [ ] `normalize_text` lowercases, strips redactions and case refs, collapses whitespace
- [ ] `tokenize_sentences` and `tokenize_words` normalise first, then tokenise
- [ ] `audit_statement` returns the correct dict shape with accurate counts
- [ ] `needs_review` is `True` only when sentence count > 4 **or** word count > 120
- [ ] `audit_archive` returns all audits sorted by `id`
- [ ] `python start.py` prints the audit table
- [ ] `pytest test_start.py -v` - 12 passed
