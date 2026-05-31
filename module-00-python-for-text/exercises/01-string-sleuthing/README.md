# Exercise 01 — String Sleuthing

The Inkwell archives are full of raw witness statements — shouted words in ALL CAPS, trailing whitespace, messy punctuation. Before any analysis can begin, you need to slice, search, and normalise these strings into something clean.

## What you practise

- String slicing and indexing
- `.lower()`, `.strip()`, `.upper()`, `.title()`
- `.split()` and `" ".join()`
- f-string formatting
- `in` membership tests and `.count()`

## Your task

Open `start.py` and implement the functions marked with `TODO`. Each function takes a raw statement string and returns a cleaned or extracted result.

## Running tests

```bash
cd module-00-python-for-text/exercises/01-string-sleuthing
pytest test_start.py -v
```

## Hints

- `str.split()` with no arguments splits on any whitespace and discards empties.
- Slicing with a negative step (`[::-1]`) reverses a string — useful for checking palindromes, less so for evidence.
- Method chaining (`text.strip().lower()`) is cleaner than temporary variables.
