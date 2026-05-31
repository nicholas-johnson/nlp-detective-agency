# Module 0 — Python for Text Analysis

> Your first day at Inkwell Investigations. The agency's archives are decades deep — witness statements, tip-off letters, newspaper clippings, and cold-case files stacked floor to ceiling. Before you can make sense of any of it, you need the analyst's toolkit: Python strings, regex, file handling, and the data structures that turn a pile of text into something you can work with.

## Learning goals

- Work fluently with Python **strings**: slicing, formatting, and common string methods.
- Use **regular expressions** to search, match, and extract patterns from text.
- Read and write text files with correct **encoding** handling (UTF-8, CSV, JSON).
- Apply **collections** (`Counter`, `defaultdict`) and **comprehensions** for text data.
- Write **type-hinted** function signatures in the style used throughout this course.
- Run scripts from the **command line** with `sys.argv` and `argparse`.
- Survey the **NLP library landscape**: NLTK, spaCy, scikit-learn, Hugging Face.

---

## Strings and slicing — reading the fine print

Witness statements are full of detail buried in long passages. Python strings give you the tools to slice out exactly what you need.

### Indexing and slicing

Every character in a string has a position (zero-indexed). Negative indices count from the end:

```python
statement = "I saw Reeves near the docks on Tuesday"
statement[0]      # 'I'
statement[-1]     # 'y'
statement[6:12]   # 'Reeves'
statement[:5]     # 'I saw'
```

Slices use the syntax `[start:stop:step]`. The `stop` index is **exclusive** — a design choice Python shares with `range()` that makes length calculations trivial: `len(s[a:b]) == b - a`.

### Common string methods

The methods you will use most in this course all return **new** strings (strings are immutable):

```python
raw = "  I saw HIM near the DOCKS on Tuesday!!!  "

raw.strip()              # remove leading/trailing whitespace
raw.lower()              # 'i saw him near the docks on tuesday!!!'
raw.upper()              # 'I SAW HIM NEAR THE DOCKS ON TUESDAY!!!'
raw.replace("!!!", ".")  # swap punctuation
raw.split()              # ['I', 'saw', 'HIM', 'near', 'the', 'DOCKS', ...]
" ".join(words)          # rejoin tokens with a single space
```

Method **chaining** is idiomatic: `raw.strip().lower().replace("!!!", "")` reads left to right, each step feeding the next.

### f-strings

Formatted string literals (f-strings) embed expressions directly. They are the standard way to build output throughout this course:

```python
witness = "Margaret Hayes"
case = "CASE-42"
print(f"Witness: {witness} | Case: {case}")
# Witness: Margaret Hayes | Case: CASE-42

count = 7
print(f"Found {count} matching statements")
```

### Encoding

Real-world text files use different encodings. **UTF-8** is the modern default and handles virtually every character. Always specify encoding explicitly when opening files to avoid platform-dependent surprises:

```python
with open("statement.txt", encoding="utf-8") as f:
    text = f.read()
```

Python 3 strings are Unicode internally, so once loaded correctly the full Unicode range (accented characters, em-dashes, curly quotes) works seamlessly.

---

## Regular expressions — pattern matching

Case files use inconsistent shorthand, redacted names, and partial dates. The `re` module lets you find patterns across thousands of documents without reading each one by hand.

### Core functions

| Function | Purpose | Returns |
| -------- | ------- | ------- |
| `re.search(pattern, text)` | Find first match anywhere | Match object or `None` |
| `re.findall(pattern, text)` | Find all non-overlapping matches | List of strings |
| `re.sub(pattern, repl, text)` | Replace every match | New string |
| `re.compile(pattern)` | Pre-compile for repeated use | Compiled pattern object |

### Practical patterns for NLP

```python
import re

statement = "[REDACTED] told me to keep quiet. CASE-42 is none of my business."

# Find case IDs like CASE-42, CASE-107
re.findall(r"CASE-\d+", statement)
# ['CASE-42']

# Find redacted placeholders
re.findall(r"\[REDACTED\]", statement)
# ['[REDACTED]']

# Strip non-alphabetic characters (keep spaces)
re.sub(r"[^a-zA-Z\s]", "", statement)
# 'REDACTED told me to keep quiet CASE is none of my business'

# Dates in YYYY-MM-DD format
text = "Recorded on 1947-03-12 at the station."
re.findall(r"\d{4}-\d{2}-\d{2}", text)
# ['1947-03-12']
```

### Flags

Flags modify matching behaviour. The most useful for text work:

```python
# Case-insensitive matching
re.findall(r"reeves", text, re.IGNORECASE)

# VERBOSE mode: spread pattern across lines with comments
date_pattern = re.compile(r"""
    \d{4}   # year
    -\d{2}  # month
    -\d{2}  # day
""", re.VERBOSE)
```

### Capture groups

Parentheses create **groups** that extract sub-matches:

```python
m = re.search(r"(CASE)-(\d+)", "CASE-42 is closed")
m.group(0)  # 'CASE-42' (full match)
m.group(1)  # 'CASE'
m.group(2)  # '42'

# findall with groups returns tuples
re.findall(r"(CASE)-(\d+)", "CASE-42 and CASE-107")
# [('CASE', '42'), ('CASE', '107')]
```

---

## File I/O and encoding — opening the archives

The filing cabinets are digital now, but the formats are a mess: plain text, CSV exports, JSON dumps. Learn to read them all without garbling a single character.

### pathlib — the modern file API

`pathlib.Path` is the standard way to work with file paths in this course. Every exercise uses the same pattern to locate the shared data directory:

```python
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
```

`Path.__file__` gives the current script's location. `.resolve()` makes it absolute. The chain of `.parent` calls walks up the directory tree to the repo root. This approach works regardless of where you run the script from.

Key `Path` operations:

```python
path = DATA_DIR / "inkwell" / "statements.json"

path.exists()        # True / False
path.read_text()     # read entire file as a string (encoding="utf-8" by default)
path.name            # 'statements.json'
path.stem            # 'statements'
path.suffix          # '.json'
```

### JSON — the course data format

All Inkwell data is stored as JSON. The standard pattern used throughout this course:

```python
import json
from pathlib import Path

path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
records = json.loads(path.read_text())
```

`json.loads()` parses a string into Python objects (lists, dicts, strings, numbers). The inverse, `json.dumps()`, serialises Python back to a JSON string:

```python
json.dumps(records[0], indent=2)
```

### CSV

Some datasets use CSV. The `csv` module in the standard library handles quoting, escaping, and delimiters:

```python
import csv

with open("data.csv", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["text"], row["label"])
```

`csv.DictReader` maps each row to a dict keyed by the header column names — convenient when you want `row["text"]` instead of `row[0]`.

---

## Collections and comprehensions — counting the evidence

How many times does a suspect's name appear? Which words show up in every statement? The right data structure makes these questions trivial.

### Lists, dicts, and sets

These three structures carry all the data in this course:

- **Lists** — ordered, mutable sequences. Use for token lists, records, results.
- **Dicts** — key-value lookup in O(1). Use for records-by-ID, frequency maps, config.
- **Sets** — unordered unique elements. Use for vocabulary, label sets, membership tests.

```python
records = json.loads(path.read_text())

# Dict for O(1) lookup by ID
by_id = {r["id"]: r for r in records}
statement = by_id["STM-003"]

# Set of unique witnesses
witnesses = {r["witness"] for r in records}
```

### List, dict, and set comprehensions

Comprehensions are the idiomatic Python way to filter, transform, and collect. Every exercise in this course uses them:

```python
# List comprehension: extract all raw_text fields
texts = [r["raw_text"] for r in records]

# With filter: only CASE-42 statements
case_42 = [r for r in records if r["case_id"] == "CASE-42"]

# Dict comprehension: map IDs to witnesses
id_to_witness = {r["id"]: r["witness"] for r in records}

# Set comprehension: unique case IDs
case_ids = {r["case_id"] for r in records}

# Nested: flatten all words across all statements
all_words = [w for r in records for w in r["raw_text"].split()]
```

### `collections.Counter`

`Counter` is a specialised dict that counts occurrences. It appears throughout this course for word frequencies, label distributions, and quick data profiling:

```python
from collections import Counter

words = "the docks the warehouse the pier seven".split()
freq = Counter(words)
# Counter({'the': 3, 'docks': 1, 'warehouse': 1, 'pier': 1, 'seven': 1})

freq.most_common(3)
# [('the', 3), ('docks', 1), ('warehouse', 1)]
```

`Counter` supports arithmetic: `counter_a + counter_b` merges counts, `counter_a - counter_b` subtracts them.

### `collections.defaultdict`

`defaultdict` provides a factory for missing keys, eliminating the need for `if key not in d` boilerplate:

```python
from collections import defaultdict

# Group statements by case
by_case = defaultdict(list)
for r in records:
    by_case[r["case_id"]].append(r)

# by_case["CASE-42"] is now a list of matching records
```

---

## Type hints — the annotation style

Every function in this course uses **type hints**. They do not change runtime behaviour, but they document the expected types for parameters and return values, catch bugs early (via editors and type checkers), and make function signatures self-documenting:

```python
def preprocess(text: str) -> list[str]:
    """Tokenise and lowercase a raw string."""
    return text.lower().split()

def filter_by_case(records: list[dict], case_id: str) -> list[dict]:
    return [r for r in records if r["case_id"] == case_id]

def find_witness(records: list[dict], name: str) -> dict | None:
    for r in records:
        if r["witness"] == name:
            return r
    return None
```

Key syntax used in this course:

| Annotation | Meaning |
| ---------- | ------- |
| `str`, `int`, `float`, `bool` | Scalar types |
| `list[str]` | List of strings |
| `dict[str, int]` | Dict mapping strings to ints |
| `set[str]` | Set of strings |
| `tuple[str, int]` | Fixed-length tuple |
| `str \| None` | String or None (Python 3.10+) |

---

## CLI basics — running scripts

### `if __name__ == "__main__":`

This guard lets a file work as both an importable module and a standalone script. When Python runs a file directly, `__name__` is set to `"__main__"`. When the file is imported, `__name__` is the module name, so the guarded code does not execute:

```python
def main():
    print("Running as a script")

if __name__ == "__main__":
    main()
```

### `sys.argv`

The simplest way to accept command-line arguments. `sys.argv` is a list of strings — the script path followed by any arguments:

```python
import sys

if len(sys.argv) > 1:
    case_id = sys.argv[1]
    print(f"Loading case {case_id}")
else:
    print("Usage: python script.py CASE-42")
```

Modules 1–5 use this lightweight approach for simple CLI arguments.

### `argparse`

For richer CLI interfaces with named flags, help text, and type conversion, `argparse` is the standard library solution. Modules 6–8 use it:

```python
import argparse

parser = argparse.ArgumentParser(description="Analyse witness statements")
parser.add_argument("case_id", help="Case identifier, e.g. CASE-42")
parser.add_argument("--limit", "-n", type=int, default=10, help="Max results")
args = parser.parse_args()

print(f"Case: {args.case_id}, limit: {args.limit}")
```

```bash
python analyse.py CASE-42 --limit 5
```

---

## The NLP toolkit — knowing your instruments

Before you preprocess a single document, survey the tools on your desk. Each library fills a different niche, and this course introduces them progressively:

| Library | What it does | First used |
| ------- | ------------ | ---------- |
| **NLTK** | Classical NLP: tokenisation, stemming, stopwords, WordNet | Module 1 |
| **spaCy** | Industrial-strength pipeline: POS tagging, NER, dependency parsing | Module 6 |
| **scikit-learn** | Machine learning: vectorisers, classifiers, evaluation metrics | Module 2 |
| **gensim** | Topic modelling (LDA) and Word2Vec training | Module 4 |
| **Hugging Face Transformers** | Pre-trained transformer models, pipelines, fine-tuning | Module 7 |

All of these are installed by `pip install -e ".[nlp,dev]"`. You do not need to install anything extra for Modules 0–6. For Module 7 (transformers, optional fine-tuning), add `pip install -e ".[nlp,local-ml,dev]"`.

---

## Field rules

- **Always specify encoding.** Default assumptions break on real-world text.
- **Regex is powerful but fragile.** Test patterns on real samples, not toy examples.
- **Use the right data structure.** Lists for sequences, dicts for lookups, Counter for frequencies.
- **Type-hint your functions.** It costs seconds and saves hours of debugging.
- **Comprehensions over loops** — when the logic fits a single filter-and-transform step, prefer a comprehension.

---

## Demos

```bash
python module-00-python-for-text/demo/01_strings_and_regex.py
python module-00-python-for-text/demo/02_collections_and_io.py
```

## Exercises

| Folder | Mission |
| ------ | ------- |
| [`exercises/01-string-sleuthing`](exercises/01-string-sleuthing/) | Slice, search, and normalise witness statement strings. |
| [`exercises/02-regex-extraction`](exercises/02-regex-extraction/) | Use regex to extract dates, case IDs, and redacted names from raw files. |
| [`exercises/03-corpus-loader`](exercises/03-corpus-loader/) | Load Inkwell JSON data with pathlib, compute word-frequency stats with Counter. |

Run tests for this module:

```bash
pytest module-00-python-for-text/
```

## Slides

From repo root: `pnpm slides:00`, or `cd module-00-python-for-text/slides && pnpm dev`.

## Reference

- [Python string methods](https://docs.python.org/3/library/stdtypes.html#string-methods)
- [re — Regular expression operations](https://docs.python.org/3/library/re.html)
- [pathlib — Object-oriented filesystem paths](https://docs.python.org/3/library/pathlib.html)
- [collections.Counter](https://docs.python.org/3/library/collections.html#collections.Counter)
- [json — JSON encoder and decoder](https://docs.python.org/3/library/json.html)
- [argparse — Parser for command-line options](https://docs.python.org/3/library/argparse.html)
