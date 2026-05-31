# Module 6 - Linguistic Structure: POS Tagging and NER

> Every statement names people, places, and dates - but they are buried in prose. "I saw him near the docks on Tuesday with a man called Reeves" contains a location, a day, and two persons of interest. POS tagging and Named Entity Recognition extract the who, where, and when so the evidence board fills itself in.

## Learning goals

- Navigate the **spaCy pipeline** from raw text to linguistic annotations.
- Apply **Part-of-Speech (POS) tagging** to understand grammatical structure.
- Read **dependency parse trees** to find subject-verb-object relationships.
- Run **Named Entity Recognition (NER)** to extract persons, organisations, locations, and dates.
- Build **custom entity rules** with spaCy's `EntityRuler`.
- Extract structured **who / where / when** records from unstructured witness statements.

## Setup

```bash
pip install -e ".[nlp,dev]"
python -m spacy download en_core_web_sm
```

This module uses **spaCy** with the small English model (`en_core_web_sm`, ~12 MB). Download once before running exercises or tests.

---

## The spaCy pipeline - the forensic linguist

spaCy processes text through a series of pipes: tokeniser → tagger → parser → NER. One call returns a rich `Doc` object with annotations at every level.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("I saw Reeves near the docks on Tuesday.")

for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)

for ent in doc.ents:
    print(ent.text, ent.label_)
```

| Object  | What it holds                                    |
| ------- | ------------------------------------------------ |
| `Doc`   | Full processed text                              |
| `Token` | One word/punctuation with POS, dependency, lemma |
| `Span`  | A slice of the doc (including NER entities)      |

Disable pipes you don't need for speed:

```python
with nlp.select_pipes(disable=["ner"]):
    doc = nlp(text)  # skips NER
```

The `en_core_web_sm` model uses a **transition-based neural pipeline** (Honnibal & Johnson, 2015) - a compact CNN-based architecture that processes tokens in context. This replaced older statistical taggers (HMM, CRF) with a single end-to-end learned model.

---

## POS tagging - grammar of the statement

Is "lead" a noun (a clue) or a verb (to guide)? **Part-of-Speech (POS) tagging** assigns a grammatical category to every token, disambiguating word roles that raw text hides.

### Tagset history

Marcus et al. (1993) created the **Penn Treebank tagset** with 36+ English-specific tags (`NN`, `NNS`, `VBD`, `JJ`, `NNP`, …). This was the standard for decades but is language-specific and fine-grained.

The **Universal Dependencies** project (Nivre et al., 2016) mapped language-specific tagsets to **17 universal POS tags (UPOS)** that work across languages:

| Tag       | Meaning                   | Example                    |
| --------- | ------------------------- | -------------------------- |
| **NOUN**  | Common noun               | docks, ledger, statement   |
| **VERB**  | Verb                      | saw, investigate, deny     |
| **ADJ**   | Adjective                 | nervous, credible, hostile |
| **ADV**   | Adverb                    | quickly, never, allegedly  |
| **PROPN** | Proper noun               | Reeves, Tuesday, London    |
| **PRON**  | Pronoun                   | he, she, they, it          |
| **DET**   | Determiner                | the, a, this, that         |
| **ADP**   | Adposition (preposition)  | near, with, at, from       |
| **NUM**   | Numeral                   | 42, three, 1987            |
| **CCONJ** | Coordinating conjunction  | and, but, or               |
| **PART**  | Particle                  | 's, not, to                |
| **INTJ**  | Interjection              | oh, yes, no                |
| **PUNCT** | Punctuation               | . , ! ?                    |
| **SYM**   | Symbol                    | $, %, @                    |
| **X**     | Other                     | foreign words, typos       |
| **AUX**   | Auxiliary verb            | is, was, have, will        |
| **SCONJ** | Subordinating conjunction | because, if, when          |

spaCy exposes both: `token.pos_` (UPOS) and `token.tag_` (Penn Treebank fine-grained).

```python
verbs = [t.text for t in doc if t.pos_ == "VERB"]
nouns = [t.text for t in doc if t.pos_ == "NOUN"]
```

Filter by POS before pattern matching - it cuts false positives sharply. If you search for location names, restrict candidates to `PROPN` and `NOUN` tokens rather than all tokens.

### Tagging algorithms (historical context)

| Approach                     | Type           | How it works                                              |
| ---------------------------- | -------------- | --------------------------------------------------------- |
| **Rule-based** (Brill, 1992) | Transformation | Start with a baseline tag, apply learned correction rules |
| **HMM**                      | Generative     | Model tag sequences as a Markov chain                     |
| **CRF**                      | Discriminative | Score entire tag sequences jointly                        |
| **Neural** (spaCy)           | Discriminative | CNN over token windows, trained end-to-end                |

You do not need to implement any of these - spaCy's pre-trained model handles tagging. Understanding the progression explains why modern pipelines are fast and accurate.

---

## Dependency parsing - who did what to whom

**Constituency parsing** (phrase-structure trees) groups words into nested phrases: `[S [NP I] [VP [V saw] [NP him]]]`. **Dependency parsing** takes a different approach: every word connects to exactly one head word, forming a tree of grammatical relationships.

Each token has:

- **`head`** - the token it depends on
- **`dep_`** - the label of the relationship

Essential relations for witness statements:

| Relation | Meaning                | Example          |
| -------- | ---------------------- | ---------------- |
| `nsubj`  | Nominal subject        | **I** → saw      |
| `dobj`   | Direct object          | saw → **him**    |
| `prep`   | Prepositional modifier | near → **docks** |
| `pobj`   | Object of preposition  | near → docks     |
| `ROOT`   | Root of the sentence   | saw              |

Nivre (2003) introduced **transition-based parsing**: read tokens left-to-right, maintain a stack and buffer, and apply shift/reduce actions to build the tree incrementally. This is how spaCy's parser works internally.

```python
for token in doc:
    if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
        verb = token.head
        obj = next((c for c in verb.children if c.dep_ == "dobj"), None)
        if obj:
            print(token.text, verb.lemma_, obj.text)
```

Pronouns (`him`), passive voice, and coordination make extraction imperfect - inspect output manually. "He was seen by Reeves" has `nsubj` = `He` but the agent is in a prepositional phrase, not the subject slot.

---

## Named Entity Recognition - naming the suspects

**Named Entity Recognition (NER)** identifies and classifies named spans in text: persons, organisations, locations, dates, and more. Unlike keyword search, NER finds multi-token entities (`Margaret Hayes`, `New York City`) and assigns type labels.

### The BIO tagging scheme

Internally, most NER models tag each token with a **BIO scheme** (Ramshaw & Marcus, 1995):

| Tag       | Meaning                            | Example              |
| --------- | ---------------------------------- | -------------------- |
| **B-PER** | Beginning of a person entity       | `Margaret`           |
| **I-PER** | Inside a person entity             | `Hayes`              |
| **B-GPE** | Beginning of a geopolitical entity | `New`                |
| **I-GPE** | Inside a GPE                       | `York`               |
| **O**     | Outside any entity                 | `saw`, `near`, `the` |

"Margaret Hayes" → `B-PER I-PER`. "New York City" → `B-GPE I-GPE I-GPE`. The model learns to predict these tags; spaCy converts them to entity spans in `doc.ents`.

```python
from collections import defaultdict

by_label = defaultdict(list)
for ent in doc.ents:
    by_label[ent.label_].append(ent.text)
```

Run over every statement in a case to build an **evidence board** - a merged list of persons, places, and dates mentioned by any witness.

### NER evaluation - span-level metrics

Evaluating NER requires **span-level** matching: a predicted entity is correct only if both the **boundaries** and the **label** match the gold standard exactly. Partial overlaps do not count.

$$\text{Precision} = \frac{\text{correct predicted spans}}{\text{all predicted spans}} \qquad \text{Recall} = \frac{\text{correct predicted spans}}{\text{all gold spans}}$$

This is stricter than token-level accuracy. Predicting `B-PER I-ORG` for "Margaret Hayes" scores zero - the boundary is right but the label is wrong.

Tjong Kim Sang and De Meulder (2003) established the **CoNLL-2003** shared task evaluation protocol that the field still uses. Exercise 02 compares spaCy NER against CoNLL gold labels using span-level precision and recall.

**NER is not perfect.** Domain names (`Reeves`, `River Lane`) may be missed or mislabelled. Always validate against source text.

---

## Custom rules - agency-specific knowledge

The pre-trained model does not know case IDs like `CASE-42` or ticket numbers like `ORD-2024-991`. Add an **EntityRuler** before the `"ner"` pipe to inject domain-specific patterns:

```python
from spacy.pipeline import EntityRuler

nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns([
    {"label": "CASE_ID", "pattern": [{"TEXT": {"REGEX": r"CASE-\d+"}}]},
    {"label": "ORDER_ID", "pattern": [
        {"TEXT": {"REGEX": r"ORD-\d{4}"}}, {"TEXT": "-"}, {"TEXT": {"REGEX": r"\d{3}"}}
    ]},
])
```

**Pattern types:**

| Type           | Format                                                | Example                                   |
| -------------- | ----------------------------------------------------- | ----------------------------------------- |
| Token pattern  | List of token attribute dicts                         | `[{"LOWER": "case"}, {"IS_DIGIT": True}]` |
| Phrase pattern | Plain string (deprecated in favour of token patterns) | `"CASE-42"`                               |

**Pipe order matters:** rules run before statistical NER so custom labels are available in the same pass. Use `overwrite_ents=True` on the ruler if you want rule matches to replace conflicting statistical NER spans.

---

## Contextual vs statistical - bridge to Module 7

spaCy's NER is a **statistical model** trained on general text (OntoNotes 5). Module 7 introduces **transformers** - contextual models that often outperform spaCy on difficult names and domain jargon, especially in zero-shot settings.

Rules and dependency walks remain useful when you need **explainability**, **offline processing**, or **hybrid pipelines** (rules catch IDs, transformers catch names).

---

## Field rules

- **NER is not perfect.** Always validate extracted entities against the source text.
- **Domain-specific names trip up general models.** Add custom rules for case-specific vocabulary.
- **Combine POS and NER.** POS filtering improves precision for pattern-based rules.
- **Inspect dependency output.** Parser errors propagate to SVO extraction - spot-check on critical sentences.

---

## Demo

Interactive console menu - explore the pipeline, SVO triples, evidence boards, and EntityRuler:

```bash
python module-06-ner-and-pos/demo/demo.py
```

---

## Exercises

| Folder                                                          | Part A (Inkwell)                        | Part B (optional `--real-world`) |
| --------------------------------------------------------------- | --------------------------------------- | -------------------------------- |
| [`exercises/01-grammar-audit`](exercises/01-grammar-audit/)     | POS + SVO triples on witness statements | SVO vs UD English EWT gold       |
| [`exercises/02-ner-extraction`](exercises/02-ner-extraction/)   | Evidence board + gold-label P/R         | NER audit on CoNLL-2003 sample   |
| [`exercises/03-custom-entities`](exercises/03-custom-entities/) | EntityRuler for `CASE-\d+`              | EntityRuler for ticket/order IDs |

Exercises are **independent** - complete them in any order.

Run an exercise interactively:

```bash
python module-06-ner-and-pos/exercises/01-grammar-audit/start.py
python module-06-ner-and-pos/exercises/02-ner-extraction/start.py
python module-06-ner-and-pos/exercises/03-custom-entities/start.py
```

Optional real-world extension:

```bash
python module-06-ner-and-pos/exercises/01-grammar-audit/start.py --real-world
python module-06-ner-and-pos/exercises/02-ner-extraction/start.py --real-world
python module-06-ner-and-pos/exercises/03-custom-entities/start.py --real-world
```

Run tests (from each exercise folder):

```bash
cd module-06-ner-and-pos/exercises/01-grammar-audit && pytest test_start.py test_extension.py -v
cd module-06-ner-and-pos/exercises/02-ner-extraction && pytest test_start.py test_extension.py -v
cd module-06-ner-and-pos/exercises/03-custom-entities && pytest test_start.py test_extension.py -v
```

## Slides

From repo root: `pnpm slides:06`, or `cd module-06-ner-and-pos/slides && pnpm dev`.

## Reference

- [spaCy - Linguistic Features](https://spacy.io/usage/linguistic-features)
- [spaCy - Named Entity Recognition](https://spacy.io/usage/linguistic-features#named-entities)
- [spaCy - Dependency Parse](https://spacy.io/usage/linguistic-features#dependency-parse)
- [spaCy - Rule-based matching](https://spacy.io/usage/rule-based-matching)
- [Universal Dependencies - English EWT](https://universaldependencies.org/treebanks/en_ewt/index.html)
- Marcus, M. P., Santorini, B., & Marcinkiewicz, M. A. (1993). Building a large annotated corpus of English. _Computational Linguistics_, 19(2), 313–330.
- Nivre, J., et al. (2016). Universal Dependencies v1. _LREC_.
- Nivre, J. (2003). An efficient algorithm for projective dependency parsing. _IWPT_, 149–160.
- Honnibal, M., & Johnson, M. (2015). An improved non-monotonic transition system for dependency parsing. _EMNLP_, 1373–1378.
- Brill, E. (1992). A simple rule-based part of speech tagger. _ANLP_, 152–155.
- Ramshaw, L., & Marcus, M. (1995). Text chunking using transformation-based learning. _WVLC_, 82–94.
- Tjong Kim Sang, E. F., & De Meulder, F. (2003). Introduction to the CoNLL-2003 shared task. _CoNLL_, 142–147.
