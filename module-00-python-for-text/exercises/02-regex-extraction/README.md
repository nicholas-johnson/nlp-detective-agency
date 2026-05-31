# Exercise 02 — Regex Extraction

Inkwell case files are riddled with inconsistent shorthand: dates in various formats, case IDs scattered through prose, redacted placeholders, and phone numbers scribbled in margins. Regular expressions let you find and extract these patterns across thousands of documents in seconds.

## What you practise

- `re.findall()` for extracting all matches
- `re.sub()` for replacing patterns
- `re.compile()` for reusable patterns
- Character classes, quantifiers, and groups
- `re.IGNORECASE` flag

## Your task

Open `start.py` and implement the functions marked with `TODO`. Each function uses regex to extract or transform specific patterns from raw case-file text.

## Running tests

```bash
cd module-00-python-for-text/exercises/02-regex-extraction
pytest test_start.py -v
```

## Hints

- `\d{4}` matches exactly four digits. `\d+` matches one or more.
- `[A-Z]` matches a single uppercase letter. `[A-Z]+` matches a run of them.
- Square brackets in the text need escaping: `\[` and `\]`.
- Use raw strings (`r"..."`) for patterns — they prevent Python from interpreting backslashes.
