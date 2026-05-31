# Exercise 03 - Custom Entities

spaCy misses agency-specific codes like `CASE-42`. Add an **EntityRuler** before the NER pipe to catch them with regex patterns.

## Before you start

```bash
pip install -e ".[nlp,dev]"
python -m spacy download en_core_web_sm
```

Open `start.py`. Each function has a `# TODO` - implement them in order.

## The data

Part A: `data/inkwell/statements.json` - statements contain `CASE-17`, `CASE-42`, `CASE-88`.

Part B: `data/public/support_tickets_sample.json` - synthetic support messages with `TKT-`, `ORD-`, and `REF-` identifiers.

## What you'll build

```bash
python start.py
```

```
Inkwell Investigations - Custom Entity Audit
================================================
CASE IDs found: 6/6 statements

Sample extractions:
  STM-001: [('CASE-42', 'CASE_ID')]
  ...
```

## Functions to implement

1. **`case_id_patterns()`** - EntityRuler patterns for `CASE-\d+`
2. **`ticket_patterns()`** - patterns for `TKT-\d+`, `ORD-\d{4}-\d+`, `REF-[A-Z0-9]+`
3. **`build_custom_nlp(patterns=None)`** - add `entity_ruler` **before** `"ner"`
4. **`extract_with_rules(nlp, texts)`** - return entity tuples per text
5. **`audit_coverage(statements)`** - count CASE IDs found vs statements containing them
6. **`audit_ticket_refs(records)`** - recall on gold refs in ticket sample

---

## Part B - Real-world extension (optional)

```bash
python start.py --real-world
```

Apply **`ticket_patterns()`** to the support tickets sample - same EntityRuler API, different domain patterns.

---

## Run the tests

```bash
cd module-06-ner-and-pos/exercises/03-custom-entities
pytest test_start.py test_extension.py -v
```

## Checklist

- [ ] EntityRuler finds `CASE-42` in STM-001
- [ ] `audit_coverage` finds nearly all case IDs
- [ ] `--real-world` recall ≥ 0.9 on ticket refs
- [ ] All tests passed
