# Exercise 03 — Corpus Loader

Every module in this course loads JSON data from the `data/` directory, computes statistics, and filters records with comprehensions. This exercise practises that exact workflow so it becomes second nature before you start Module 1.

## What you practise

- `pathlib.Path` for locating files
- `json.loads()` for parsing JSON
- `collections.Counter` for word frequencies
- List and dict comprehensions for filtering and transforming
- `sys.argv` for a simple CLI argument

## Your task

Open `start.py` and implement the functions marked with `TODO`. The functions load Inkwell data, compute word-frequency statistics, and filter records — the same patterns you will use throughout this course.

## Running tests

```bash
cd module-00-python-for-text/exercises/03-corpus-loader
pytest test_start.py -v
```

## Running as a CLI

```bash
python start.py                  # prints summary of all statements
python start.py CASE-42          # filters to a specific case
```

## Hints

- The data path pattern: `Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"`
- `Counter.most_common(n)` returns the top *n* items as `[(word, count), ...]`.
- `defaultdict(list)` lets you group records by key without checking `if key in dict`.
