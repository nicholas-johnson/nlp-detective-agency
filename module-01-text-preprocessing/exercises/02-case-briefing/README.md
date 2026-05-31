# Exercise 02 - Case Briefing

Detectives working a single case need a word-frequency briefing - what themes keep appearing across witness statements? Your job: preprocess the text, strip noise, lemmatise the tokens, and produce a ranked list of the most common terms for one case.

This exercise builds on the cleaning ideas from Exercise 01 but adds **stopwords**, **lemmatisation**, and **aggregation**. The deliverable is different: a thematic briefing for one `case_id`, not a per-statement audit.

## Before you start

Complete Exercise 01 first (or at least read its `normalize_text` logic). You will reimplement `normalize_text` in this file - exercises are self-contained and do not import from each other.

NLTK setup (if you have not already):

```bash
pip install -e ".[nlp,dev]"
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

Open `start.py` in this folder. The module-level constants `STOPS` and `LEMMATIZER` are already set up for you.

## The data

Same archive as Exercise 01: `data/inkwell/statements.json`. Statements are tagged with a `case_id` (`CASE-17`, `CASE-42`, `CASE-88`). For this exercise you filter to one case and summarise its vocabulary.

There are **4 statements** for `CASE-42` (the docks case).

## What you'll build

When your functions are complete:

```bash
python start.py CASE-42
```

prints something like:

```
Inkwell Investigations - Briefing for CASE-42
========================================
   1. dock                 4
   2. reef                 4
   3. warehouse            3
   4. saw                  2
   ...
```

(Exact rankings may vary slightly depending on lemmatisation - `docks` becomes `dock`, `reeves` may become `reef`.)

If you pass an unknown case ID, you should see `No statements found for that case.`

The `main()` function is already written - you implement the pipeline functions it relies on.

## The preprocessing pipeline

Each statement goes through these steps in order:

```
raw_text → normalize_text → word_tokenize → remove_stopwords → lemmatize_tokens
```

Only tokens with **length >= 3** are counted in the final frequency table.

## Functions to implement

### 1. `normalize_text(text)`

Same rules as Exercise 01:

1. Lowercase
2. Remove `[REDACTED]` (case-insensitive)
3. Remove `CASE-XXX` references (`case-\d+`)
4. Collapse whitespace

```python
text = text.lower()
text = re.sub(r"\[redacted\]", " ", text, flags=re.IGNORECASE)
text = re.sub(r"case-\d+", " ", text, flags=re.IGNORECASE)
text = re.sub(r"\s+", " ", text).strip()
```

### 2. `remove_stopwords(tokens)`

Return only **alphabetic** tokens that are **not** in `STOPS` (already defined at the top of `start.py`):

```python
return [t for t in tokens if t.isalpha() and t not in STOPS]
```

### 3. `lemmatize_tokens(tokens)`

Lemmatise each alphabetic token using `LEMMATIZER` (WordNet lemmatiser, default noun POS):

```python
return [LEMMATIZER.lemmatize(t) for t in tokens if t.isalpha()]
```

This collapses word forms: `docks` → `dock`, `investigating` → `investigating` (noun default). That is fine for this exercise.

### 4. `preprocess_statement(text)`

Chain the steps together and return a list of lemma tokens:

```python
def preprocess_statement(text: str) -> list[str]:
    text = normalize_text(text)
    tokens = word_tokenize(text)
    tokens = remove_stopwords(tokens)
    return lemmatize_tokens(tokens)
```

### 5. `statements_for_case(statements, case_id)`

Filter the archive to one case:

```python
return [s for s in statements if s["case_id"] == case_id]
```

### 6. `term_frequencies(texts)`

Count preprocessed tokens across multiple raw text strings. Use `collections.Counter`:

```python
counter: Counter[str] = Counter()
for text in texts:
    for token in preprocess_statement(text):
        if len(token) >= 3:
            counter[token] += 1
return counter
```

Only count tokens with **length >= 3** - this filters out short noise like `am`, `ok`.

### 7. `case_briefing(statements, case_id, top_n=10)`

Put it together:

1. Filter statements with `statements_for_case`
2. If none found, return `[]`
3. Extract `raw_text` from each matching statement
4. Call `term_frequencies` on those texts
5. Return `.most_common(top_n)` - a list of `(term, count)` tuples, highest count first

```python
case_stmts = statements_for_case(statements, case_id)
if not case_stmts:
    return []
texts = [s["raw_text"] for s in case_stmts]
return term_frequencies(texts).most_common(top_n)
```

---

## Run it

From the repo root:

```bash
python module-01-text-preprocessing/exercises/02-case-briefing/start.py CASE-42
```

Try other cases:

```bash
python module-01-text-preprocessing/exercises/02-case-briefing/start.py CASE-17
python module-01-text-preprocessing/exercises/02-case-briefing/start.py CASE-99   # no statements
```

Without a command-line argument, `main()` prompts you for a case ID (defaults to `CASE-42` if you press Enter on an empty line).

## Run the tests

Run pytest **from this exercise folder**:

```bash
cd module-01-text-preprocessing/exercises/02-case-briefing
pytest test_start.py -v
```

All 9 tests should pass when you are done.

---

## Optional hints

<details>
<summary>Hint: why reimplement normalize_text?</summary>

Each exercise is self-contained so you can copy or zip a single folder without broken imports. Copy your working `normalize_text` from Exercise 01 - the logic is identical.

</details>

<details>
<summary>Hint: stopwords are already loaded</summary>

`STOPS = set(stopwords.words("english"))` runs at import time. If you see a `LookupError` for stopwords:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

</details>

<details>
<summary>Hint: alphabetic tokens only</summary>

Both `remove_stopwords` and `lemmatize_tokens` should skip non-alphabetic tokens (`token.isalpha()`). Punctuation tokens from `word_tokenize` should not reach the counter.

</details>

<details>
<summary>Hint: Counter.most_common</summary>

`Counter` is a dict subclass with a handy method:

```python
from collections import Counter

counts = Counter(["dock", "dock", "warehouse", "dock"])
counts.most_common(2)   # [('dock', 3), ('warehouse', 1)]
```

It returns `(item, count)` pairs sorted by count descending.

</details>

<details>
<summary>Hint: empty case returns empty list</summary>

If `statements_for_case` finds no matches (e.g. `CASE-99`), return `[]` immediately. Do not call `most_common` on an empty counter - just return `[]`.

</details>

<details>
<summary>Hint: lemmatisation quirks with proper nouns</summary>

WordNet lemmatisation can mangle names. `reeves` may lemmatise to `reef` because the lemmatiser assumes it is a plural noun. That is expected with default settings. The tests check for `dock` and `warehouse`, not `reeves`.

</details>

<details>
<summary>Hint: difference from the demo archive briefing</summary>

The module demo (`demo/demo.py`) computes top terms across **all** cases. This exercise filters to **one** `case_id` - a narrower, case-specific briefing. Same techniques, different scope.

</details>

---

## How this differs from Exercise 01

|               | Exercise 01          | Exercise 02        |
| ------------- | -------------------- | ------------------ |
| Goal          | Audit each statement | Summarise one case |
| Tokenisation  | Sentences + words    | Words only         |
| Stopwords     | No                   | Yes                |
| Lemmatisation | No                   | Yes                |
| Output        | Per-statement dict   | Ranked term list   |
| Scope         | Entire archive       | Single `case_id`   |

---

## Checklist

- [ ] `normalize_text` matches Exercise 01 cleaning rules
- [ ] `remove_stopwords` keeps only alphabetic, non-stopword tokens
- [ ] `lemmatize_tokens` lemmatises alphabetic tokens
- [ ] `preprocess_statement` chains all four steps in order
- [ ] `statements_for_case` filters by exact `case_id` match
- [ ] `term_frequencies` counts only tokens with `len >= 3`
- [ ] `case_briefing` returns `[]` for unknown cases
- [ ] `case_briefing` returns `top_n` `(term, count)` pairs sorted by frequency
- [ ] `python start.py CASE-42` prints a briefing
- [ ] `pytest test_start.py -v` - 9 passed
