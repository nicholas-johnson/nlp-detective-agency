# Module 6 — Linguistic Structure: POS Tagging and NER

> Every statement names people, places, and dates — but they are buried in prose. "I saw him near the docks on Tuesday with a man called Reeves" contains a location, a day, and two persons of interest. POS tagging and Named Entity Recognition extract the who, where, and when so the evidence board fills itself in.

## Learning goals

- Navigate the **spaCy pipeline** from raw text to linguistic annotations.
- Apply **Part-of-Speech (POS) tagging** to understand grammatical structure.
- Read **dependency parse trees** to find subject-verb-object relationships.
- Run **Named Entity Recognition (NER)** to extract persons, organisations, locations, and dates.
- Build **custom entity rules** with spaCy's `Matcher` and `EntityRuler`.
- Extract structured **who / where / when** records from unstructured witness statements.

---

## The spaCy pipeline — the forensic linguist

spaCy processes text through a series of pipes: tokeniser, tagger, parser, NER. Understand what each stage produces and how to access it.

<!-- Skeleton: nlp() call, Doc/Token/Span objects, displacy visualisation -->

---

## POS tagging — grammar of the statement

Is "lead" a noun (a clue) or a verb (to guide)? POS tags disambiguate word roles and underpin more advanced extraction.

<!-- Skeleton: universal POS tags, token.pos_, token.tag_, filtering by POS -->

---

## Dependency parsing — who did what to whom

Dependency trees reveal grammatical relationships: who is the subject, what is the object, which words modify which. Essential for understanding witness statements.

<!-- Skeleton: token.dep_, token.head, walking the tree, displacy dependency visualisation -->

---

## Named Entity Recognition — naming the suspects

NER labels spans of text as PERSON, ORG, GPE, DATE, and more. Run it over every statement and compile a suspect list automatically.

<!-- Skeleton: ent.label_, ent.text, standard entity types, batch processing a corpus -->

---

## Custom rules — agency-specific knowledge

The pre-trained model does not know your case-specific codes. EntityRuler and Matcher let you add patterns for case IDs, informant handles, and agency jargon.

<!-- Skeleton: EntityRuler patterns, Matcher with token patterns, combining with statistical NER -->

---

## Field rules

- **NER is not perfect.** Always validate extracted entities against the source text.
- **Domain-specific names trip up general models.** Add custom rules for case-specific vocabulary.
- **Combine POS and NER.** POS filtering can improve extraction precision for pattern-based rules.

---

## Exercises

| Folder | Mission |
| ------ | ------- |
| [`exercises/01-pos-tagging`](exercises/01-pos-tagging/) | POS-tag witness statements and filter by grammatical role. |
| [`exercises/02-ner-extraction`](exercises/02-ner-extraction/) | Extract persons, locations, and dates from case files with spaCy NER. |
| [`exercises/03-custom-entities`](exercises/03-custom-entities/) | Add EntityRuler patterns for case IDs and informant aliases. |

Run tests for this module:

```bash
pytest module-06-ner-and-pos/
```

## Slides

From repo root: `pnpm slides:06`, or `cd module-06-ner-and-pos/slides && pnpm dev`.

## Reference

- [spaCy — Named Entity Recognition](https://spacy.io/usage/linguistic-features#named-entities)
- [spaCy — Rule-based matching](https://spacy.io/usage/rule-based-matching)
- [spaCy — Dependency Parse](https://spacy.io/usage/linguistic-features#dependency-parse)
