# Module 0 — Python for Text Analysis

> Your first day at Inkwell Investigations. The agency's archives are decades deep — witness statements, tip-off letters, newspaper clippings, and cold-case files stacked floor to ceiling. Before you can make sense of any of it, you need the analyst's toolkit: Python strings, regex, file handling, and the data structures that turn a pile of text into something you can work with.

## Learning goals

- Work fluently with Python **strings**: slicing, formatting, and common string methods.
- Use **regular expressions** to search, match, and extract patterns from text.
- Read and write text files with correct **encoding** handling (UTF-8, CSV, JSON).
- Apply **collections** (`Counter`, `defaultdict`) and **comprehensions** for text data.
- Load and inspect tabular text data with **pandas** basics.
- Survey the **NLP library landscape**: NLTK, spaCy, scikit-learn, Hugging Face.

---

## Strings and slicing — reading the fine print

Witness statements are full of detail buried in long passages. Python strings give you the tools to slice out exactly what you need.

<!-- Skeleton: string methods, slicing, f-strings, encoding basics -->

---

## Regular expressions — pattern matching

Case files use inconsistent shorthand, redacted names, and partial dates. Regex lets you find patterns across thousands of documents without reading each one by hand.

<!-- Skeleton: re module, match/search/findall, common patterns for dates, names, phone numbers -->

---

## File I/O and encoding — opening the archives

The filing cabinets are digital now, but the formats are a mess: plain text, CSV exports, JSON dumps. Learn to read them all without garbling a single character.

<!-- Skeleton: Path, open(), encoding=, json.loads, csv.reader, pandas read_csv -->

---

## Collections and comprehensions — counting the evidence

How many times does a suspect's name appear? Which words show up in every statement? Counters and comprehensions make tallies trivial.

<!-- Skeleton: Counter, defaultdict, list/dict/set comprehensions applied to token lists -->

---

## The NLP toolkit — knowing your instruments

Before you preprocess a single document, survey the tools on your desk: NLTK for classical NLP, spaCy for production pipelines, scikit-learn for machine learning, Hugging Face for transformers.

<!-- Skeleton: install overview, when to reach for each library, environment setup -->

---

## Field rules

- **Always specify encoding.** Default assumptions break on real-world text.
- **Regex is powerful but fragile.** Test patterns on real samples, not toy examples.
- **Use the right data structure.** Lists for sequences, dicts for lookups, Counter for frequencies.

---

## Exercises

| Folder | Mission |
| ------ | ------- |
| [`exercises/01-string-sleuthing`](exercises/01-string-sleuthing/) | Slice, search, and transform witness statement strings. |
| [`exercises/02-regex-extraction`](exercises/02-regex-extraction/) | Use regex to extract dates, names, and case IDs from raw files. |
| [`exercises/03-corpus-loader`](exercises/03-corpus-loader/) | Load a tabular case-file dataset with pandas and compute basic statistics. |

Run tests for this module:

```bash
pytest module-00-python-for-text/
```

## Slides

From repo root: `pnpm slides:00`, or `cd module-00-python-for-text/slides && pnpm dev`.

## Reference

- [Python string methods](https://docs.python.org/3/library/stdtypes.html#string-methods)
- [re — Regular expression operations](https://docs.python.org/3/library/re.html)
- [collections.Counter](https://docs.python.org/3/library/collections.html#collections.Counter)
- [pandas — Getting started](https://pandas.pydata.org/docs/getting_started/index.html)
