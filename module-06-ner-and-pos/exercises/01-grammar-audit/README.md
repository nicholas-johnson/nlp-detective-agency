# Exercise 01 - Grammar Audit

POS-tag witness statements and extract **subject-verb-object** triples via dependency parsing. Understand who did what to whom before reading entity names.

## Before you start

```bash
pip install -e ".[nlp,dev]"
python -m spacy download en_core_web_sm
```

Open `start.py`. Each function has a `# TODO` - implement them in order.

## The data

Witness statements in `data/inkwell/statements.json` - filter by `case_id` (default audit: `CASE-42`).

## What you'll build

```bash
python start.py
```

```
Inkwell Investigations - Grammar Audit (CASE-42)
================================================

STM-001 Margaret Hayes:
  Verbs (4): saw, was, called, told
  SVO: (I, see, him)
  ...
```

## Functions to implement

1. **`load_nlp()`** - load `en_core_web_sm`, cache in `_nlp`
2. **`load_statements(path, case_id=None)`** - load JSON, optional filter
3. **`pos_summary(doc)`** - return `{verb_count, noun_count, verbs, nouns}`
4. **`extract_svo_triples(doc)`** - walk `nsubj`/`nsubjpass` → verb → `dobj`/`attr`; verb = lemma
5. **`audit_case(statements, case_id)`** - combine pos + SVO per statement
6. **`score_svo(predicted, gold)`** - flexible match; return `{matched, total, recall}`

---

## Part B - Real-world extension (optional)

```bash
python start.py --real-world
```

Score SVO extraction against gold triples in `data/public/ud_ewt_sample.json` (UD English EWT, CC BY-SA).

Implement **`load_ud_sample(path)`** and reuse `score_svo()`.

---

## Run the tests

```bash
cd module-06-ner-and-pos/exercises/01-grammar-audit
pytest test_start.py test_extension.py -v
```

## Checklist

- [ ] `extract_svo_triples` finds at least one triple in "She saw him near the station."
- [ ] `audit_case` returns results for all CASE-42 statements
- [ ] `--real-world` prints overall SVO recall
- [ ] All tests passed
