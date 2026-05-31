# Exercise 02 - NER Extraction

Run spaCy NER over witness statements and build a per-case **evidence board** - merged persons, locations, and dates. Score against gold labels.

## Before you start

```bash
pip install -e ".[nlp,dev]"
python -m spacy download en_core_web_sm
```

Open `start.py`. Each function has a `# TODO` - implement them in order.

## The data

| File                                   | Purpose                        |
| -------------------------------------- | ------------------------------ |
| `data/inkwell/statements.json`         | Witness statements             |
| `data/inkwell/statement_entities.json` | Gold NER labels for evaluation |

Gold labels use coarse types: `PERSON`, `LOC`, `DATE`. Map spaCy `GPE`/`FAC` → `LOC` in your evaluator.

## What you'll build

```bash
python start.py
```

```
Inkwell Investigations - NER Evidence Board (CASE-42)
====================================================
DATE: Tuesday, midnight
LOC: Anchor, docks, pier seven, warehouse
PERSON: Margaret Hayes, Marsh, Reeves, Whitfield

Gold-label evaluation (sample statements):
  DATE     P=0.500 R=0.500 (n=2)
  LOC      P=0.667 R=0.600 (n=5)
  PERSON   P=0.800 R=0.750 (n=4)
```

## Functions to implement

1. **`load_nlp()`** - load and cache spaCy model
2. **`extract_entities(doc)`** - group `doc.ents` by label
3. **`build_evidence_board(statements, case_id)`** - merge entity sets across witnesses
4. **`compare_witnesses(statements, case_id)`** - per-witness entity dicts
5. **`evaluate_ner(predicted, gold)`** - per-label precision/recall; items are `{text, label}`

Use flexible text matching (case-insensitive substring) for evaluation.

---

## Part B - Real-world extension (optional)

```bash
python start.py --real-world
```

Evaluate NER on `data/public/conll_ner_sample.json` (~40 sentences from CoNLL-2003). Same `evaluate_ner()` function.

---

## Run the tests

```bash
cd module-06-ner-and-pos/exercises/02-ner-extraction
pytest test_start.py test_extension.py -v
```

## Checklist

- [ ] Evidence board for CASE-42 includes Reeves or docks
- [ ] `evaluate_ner` returns precision/recall per label
- [ ] `--real-world` prints CoNLL metrics
- [ ] All tests passed
