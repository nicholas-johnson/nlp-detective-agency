export const slides = [
  {
    type: 'title',
    content: {
      title: 'Python NLP',
      subtitle: 'Inkwell Investigations - Nicholas Johnson',
      icon: 'search',
    },
  },
  {
    type: 'title',
    content: {
      title: 'Module 6 - Linguistic Structure',
      subtitle: 'POS tagging, dependencies, and NER with spaCy',
      icon: 'file-text',
    },
  },
  {
    type: 'welcome',
    content: {
      title: 'Who, where, when - buried in prose',
      points: [
        '"I saw **him** near the **docks** on **Tuesday** with a man called **Reeves**."',
        'Names, places, and dates hide in witness statements.',
        'spaCy adds **POS tags**, **dependency trees**, and **NER** in one pipeline.',
        'Module 7 replaces much of this with **transformers** - rules and deps stay useful.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Learning goals',
      icon: 'target',
      points: [
        'Understand **POS tagging** and **NER** as core NLP tasks.',
        'Navigate the **spaCy pipeline** - tokenizer, tagger, parser, NER.',
        'Extract **SVO triples** - who did what to whom.',
        'Build an **evidence board** with NER.',
        'Extend with **EntityRuler** for case IDs and ticket refs.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'POS tagging',
      subtitle: 'Grammar of the statement',
      icon: 'type',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'What is POS tagging?',
      icon: 'help-circle',
      points: [
        'Labels every word with its **grammatical role**: noun, verb, adjective, ...',
        'Is "lead" a **noun** (a clue) or a **verb** (to guide)? POS disambiguates.',
        'Filter for **verbs** to find actions, **nouns** to find subjects and objects.',
        'Enables **dependency parsing** and **SVO extraction** downstream.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Universal POS tags',
      icon: 'tag',
      points: [
        '17 language-independent categories from the **Universal Dependencies** project.',
        'VERB, NOUN, PROPN, ADJ, ADV, PRON, DET, ADP, NUM, …',
        'Fine-grained tags add detail: VBD (past-tense verb), NNP (proper noun singular).',
        '**Lemmatisation** maps inflected forms to dictionary roots (`saw` → `see`).',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Named Entity Recognition',
      subtitle: 'Naming the suspects',
      icon: 'users',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'What is NER?',
      icon: 'help-circle',
      points: [
        'Finds and classifies **proper names, places, dates, organisations** in running text.',
        'Works at the **span level** — "Margaret Hayes" is one entity, not two words.',
        'Sequence labelling: each token gets a tag — **B-PERSON** (begin), **I-PERSON** (inside), **O** (outside).',
        'General models miss domain-specific IDs — we fix that with custom rules later.',
      ],
    },
  },
  {
    type: 'cards',
    content: {
      title: 'Common entity types',
      cards: [
        { heading: 'PERSON', body: 'Margaret Hayes, Reeves' },
        { heading: 'GPE / LOC', body: 'docks, River Lane' },
        { heading: 'DATE / TIME', body: 'Tuesday, midnight' },
        { heading: 'ORG', body: 'Agencies, companies' },
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'The spaCy pipeline',
      subtitle: 'POS, deps, and NER in one call',
      icon: 'layers',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'What is spaCy?',
      icon: 'package',
      points: [
        'Not an acronym — named by creator **Matthew Honnibal** (stylised as spaCy).',
        'An open-source **industrial-strength** NLP library in Python.',
        'Pre-trained models for 70+ languages — we use `en_core_web_sm` (~12 MB).',
        'One call to `nlp(text)` runs **tokenization, POS, parsing, and NER** in sequence.',
        'Fast: CNN-based pipeline processes thousands of documents per second.',
        'Extensible: add custom pipes (like **EntityRuler**) anywhere in the chain.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'Dependency tree — "I saw Reeves near the docks"',
      code: `        saw (ROOT)
       / \\
      I    Reeves    near
   (nsubj)  (dobj)   (prep)
                        \\
                       docks
                       (pobj)`,
      highlights: [
        'Every word points to its **head** — the word it depends on.',
        'The **root** verb anchors the tree; subjects and objects hang off it.',
        'Walk `nsubj` → verb → `dobj` to extract **SVO triples**.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Doc, Token, Span',
      icon: 'code',
      points: [
        '`nlp = spacy.load("en_core_web_sm")` — load a pre-trained pipeline.',
        '`doc = nlp(text)` — tokenizer → tagger → parser → NER in one pass.',
        '`Token` — text, `pos_`, `dep_`, `head`, `lemma_`.',
        '`Span` — slice including **NER entities** (`doc.ents`).',
        'Disable pipes you don\'t need: `nlp.select_pipes(disable=["ner"])`.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Pipelines in practice',
      subtitle: 'What can you build?',
      icon: 'route',
    },
  },
  {
    type: 'cards',
    content: {
      title: 'Real-world NLP pipelines',
      cards: [
        {
          heading: 'News monitoring',
          body: 'Extract **persons**, **orgs**, and **locations** from articles to track who is mentioned where.',
        },
        {
          heading: 'Legal discovery',
          body: 'POS-filter for **verbs** of obligation ("shall", "must") and NER for **parties** and **dates** in contracts.',
        },
        {
          heading: 'Medical records',
          body: 'Custom EntityRuler catches **drug codes** and **dosages**; NER finds **conditions** and **providers**.',
        },
        {
          heading: 'Customer support',
          body: 'SVO triples surface **complaints** ("product broke"); NER extracts **order IDs** and **product names**.',
        },
      ],
    },
  },

  {
    type: 'code',
    content: {
      title: 'POS filtering — find the verbs',
      code: `import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("She denied meeting Reeves at the warehouse.")

verbs = [t.text for t in doc if t.pos_ == "VERB"]
nouns = [t.text for t in doc if t.pos_ == "NOUN"]

# verbs → ['denied', 'meeting']
# nouns → ['warehouse']`,
      highlights: [
        '`token.pos_` gives the **Universal POS tag** for every token.',
        'Filter before pattern matching — cuts false positives sharply.',
        '`token.lemma_` normalises tense: `denied` → `deny`.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'NER — extract names, places, dates',
      code: `doc = nlp("Margaret Hayes saw Reeves near the docks on Tuesday.")

for ent in doc.ents:
    print(ent.text, ent.label_)

# Margaret Hayes  PERSON
# Reeves          PERSON
# Tuesday         DATE`,
      highlights: [
        '`doc.ents` contains **Span** objects with `.text` and `.label_`.',
        'Multi-word names like **Margaret Hayes** are a single entity.',
        'The model misses domain IDs — that is what EntityRuler fixes.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'SVO extraction — walk the dependency tree',
      code: `doc = nlp("I saw Reeves near the docks.")

for token in doc:
    if token.dep_ == "nsubj":
        verb = token.head
        obj = next(
            (c for c in verb.children if c.dep_ == "dobj"),
            None
        )
        if obj:
            print(token.text, verb.lemma_, obj.text)

# I  see  Reeves`,
      highlights: [
        'Find `nsubj` → follow `.head` to the verb → find `dobj` child.',
        'Verb stored as **lemma** so tenses match (`saw` → `see`).',
        'This is the core of Exercise 01.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Demo - Pipeline tour',
      subtitle:
        'python module-06-ner-and-pos/demo/demo.py - option 1',
      icon: 'terminal',
    },
  },

  {
    type: 'title',
    content: {
      title: 'Dependency parsing',
      subtitle: 'Who did what to whom',
      icon: 'git-branch',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'What is a dependency tree?',
      icon: 'help-circle',
      points: [
        'Every word points to a **parent** — its grammatical head.',
        'The **root** is usually the main verb; subjects and objects hang off it.',
        'The tree captures *who did what to whom* structurally.',
        'Walk the tree programmatically: `token.head`, `token.children`.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Key relations',
      icon: 'link',
      points: [
        '`nsubj` - nominal **subject** linked to verb.',
        '`dobj` - direct **object** of verb.',
        '`token.head` - syntactic parent; `token.children` - dependents.',
        'Walk: subject → verb lemma → object.',
        'Imperfect on pronouns and passive voice - inspect output.',
      ],
    },
  },

  {
    type: 'standard',
    content: {
      title: 'What are SVO triples?',
      icon: 'list',
      points: [
        'A triple is a **(subject, verb, object)** fact extracted from a sentence.',
        '"I saw Reeves" → **(I, see, Reeves)** — verb stored as its lemma.',
        'Walk the dependency tree: `nsubj` → head verb → `dobj`.',
        'Reduces messy prose to a **structured table** of who did what to whom.',
        'Imperfect on pronouns, passive voice, and coordination — inspect manually.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Demo - SVO triples by case',
      subtitle:
        'python module-06-ner-and-pos/demo/demo.py - option 2',
      icon: 'terminal',
    },
  },

  {
    type: 'title',
    content: {
      title: 'The Evidence Board',
      subtitle: 'NER across an entire case',
      icon: 'clipboard',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Evidence board',
      icon: 'clipboard',
      points: [
        'Run NER on every statement in a **case**.',
        'Merge entities across witnesses into one board.',
        "Compare witnesses - who mentions Reeves? Who doesn't?",
        'Score against **gold labels** - NER is never perfect.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Demo - NER evidence board and displacy',
      subtitle:
        'python module-06-ner-and-pos/demo/demo.py - options 3 & 5',
      icon: 'terminal',
    },
  },

  {
    type: 'title',
    content: {
      title: 'Custom rules',
      subtitle: 'EntityRuler',
      icon: 'settings',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Why custom rules?',
      icon: 'help-circle',
      points: [
        'Pre-trained NER knows **common** entities — persons, cities, dates.',
        'It has **never seen** your domain IDs: `CASE-42`, `TKT-8842`, `ORD-2024-991`.',
        'These patterns are **regular** — regex catches them perfectly.',
        'spaCy\'s **EntityRuler** injects rule-based matches into the same pipeline.',
        'Rules for IDs + NER for names = **hybrid pipeline** with best of both.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Patterns before statistical NER',
      icon: 'regex',
      points: [
        '`nlp.add_pipe("entity_ruler", before="ner")` — insert **before** statistical NER.',
        'Ruler claims spans first; NER fills in the rest.',
        'Two pattern types: **token patterns** (attribute dicts) and **regex** on token text.',
        'Pipe **order matters** — `before="ner"` is the key.',
      ],
    },
  },
  {
    type: 'cards',
    content: {
      title: 'Domain patterns you might add',
      cards: [
        {
          heading: 'Case IDs',
          body: '`CASE-\\d+` → CASE-42, CASE-17, CASE-88',
        },
        {
          heading: 'Support tickets',
          body: '`TKT-\\d+` → TKT-8842, TKT-1001',
        },
        {
          heading: 'Order refs',
          body: '`ORD-\\d{4}-\\d+` → ORD-2024-991 (may span multiple tokens)',
        },
        {
          heading: 'Reference codes',
          body: '`REF-[A-Z0-9]+` → REF-AB12, REF-X9K3',
        },
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'EntityRuler — single-token pattern',
      code: `import spacy

nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler", before="ner")

ruler.add_patterns([
    {"label": "CASE_ID",
     "pattern": [{"TEXT": {"REGEX": r"CASE-\\d+"}}]},
])

doc = nlp("Refer to CASE-42 for details.")
for ent in doc.ents:
    print(ent.text, ent.label_)

# CASE-42  CASE_ID`,
      highlights: [
        'Pattern is a **list of token dicts** — one dict per token to match.',
        '`TEXT` with `REGEX` matches the token\'s exact text against a regex.',
        'Ruler runs **before** NER, so `CASE-42` is claimed before the model sees it.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'EntityRuler — multi-token pattern',
      code: `ruler.add_patterns([
    {"label": "ORDER_ID", "pattern": [
        {"TEXT": {"REGEX": r"ORD-\\d{4}"}},
        {"TEXT": "-"},
        {"TEXT": {"REGEX": r"\\d{3}"}},
    ]},
])

doc = nlp("See order ORD-2024-991 for refund.")
for ent in doc.ents:
    print(ent.text, ent.label_)

# ORD-2024-991  ORDER_ID`,
      highlights: [
        'The tokenizer splits `ORD-2024-991` into **three tokens**: `ORD-2024`, `-`, `991`.',
        'Each dict matches **one token** — the list matches a contiguous span.',
        'Always check how spaCy **tokenizes** your IDs before writing patterns.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Demo - EntityRuler for CASE IDs',
      subtitle:
        'python module-06-ner-and-pos/demo/demo.py - option 4',
      icon: 'terminal',
    },
  },

  {
    type: 'title',
    content: {
      title: 'Exercises',
      subtitle: 'Independent - any order',
      icon: 'book-open',
    },
  },
  {
    type: 'cards',
    content: {
      title: 'Three missions',
      cards: [
        {
          heading: '01 Grammar audit',
          body: 'POS + SVO triples on Inkwell. Part B: UD English sample.',
        },
        {
          heading: '02 NER extraction',
          body: 'Evidence board + gold eval. Part B: CoNLL-2003 sample.',
        },
        {
          heading: '03 Custom entities',
          body: 'EntityRuler for CASE IDs. Part B: support ticket refs.',
        },
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Part B - real-world extensions',
      icon: 'globe',
      points: [
        'Optional: uncomment `run_real_world()` in `main()`',
        'Same functions, public datasets under `data/public/`.',
        'Expect **lower scores** on real data - domain shift is normal.',
        'Core tests use Inkwell only.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Field rules',
      subtitle: 'Before you ship',
      icon: 'alert-triangle',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Practical guidance',
      icon: 'check-circle',
      points: [
        '**Validate NER** against source text - never trust blindly.',
        '**Add rules** for domain IDs the model misses.',
        '**Inspect dependencies** on critical sentences.',
        '**Hybrid pipelines** - rules for IDs, NER for names.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Bridge to Module 7',
      subtitle: 'Transformers and contextual NER',
      icon: 'arrow-right',
    },
  },
  {
    type: 'welcome',
    content: {
      title: 'What comes next',
      points: [
        'Module 7 - **Hugging Face pipelines** for NER, sentiment, QA.',
        'Contextual embeddings resolve ambiguity BoW and static vectors miss.',
        'Keep spaCy skills for **offline**, **explainable**, **rule-based** layers.',
        'Module 8 chains everything into an end-to-end capstone.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Setup reminder',
      icon: 'download',
      points: [
        '`pip install -e ".[nlp,dev]"`',
        '`python -m spacy download en_core_web_sm`',
        '`python module-06-ner-and-pos/demo/demo.py`',
        '`pnpm slides:06`',
      ],
    },
  },
];
