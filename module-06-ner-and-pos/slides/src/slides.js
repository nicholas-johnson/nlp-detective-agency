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
        'Navigate the **spaCy pipeline** - tokenizer, tagger, parser, NER.',
        'Filter by **POS** and walk **dependency** relations.',
        'Extract **SVO triples** - who did what to whom.',
        'Build an **evidence board** with NER.',
        'Extend with **EntityRuler** for case IDs and ticket refs.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'The spaCy pipeline',
      subtitle: 'One call, rich annotations',
      icon: 'layers',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Doc, Token, Span',
      icon: 'code',
      points: [
        '`nlp = spacy.load("en_core_web_sm")`',
        '`doc = nlp(text)` - processed document.',
        '`Token` - text, `pos_`, `dep_`, `head`, `lemma_`.',
        '`Span` - slice including **NER entities** (`doc.ents`).',
        'Disable pipes: `nlp.select_pipes(disable=["ner"])`.',
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
        'Why: filter for **verbs** to find actions, **nouns** to find subjects and objects.',
        'Enables **dependency parsing** and **SVO extraction** downstream.',
        'spaCy does it automatically as part of `nlp(text)` — no extra step needed.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Universal POS tags',
      icon: 'tag',
      points: [
        '`token.pos_` - VERB, NOUN, PROPN, ADJ, …',
        '`token.tag_` - fine-grained English tag (VBD, NNP).',
        '`token.lemma_` - dictionary form (`saw` → `see`).',
        'Filter verbs before SVO extraction; filter nouns for keyword lists.',
      ],
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
        'Sequence labelling: each token gets a tag — **B-PERSON** (begin), **I-PERSON** (inside), **O** (outside).',
        'Pre-trained spaCy model handles common entities out of the box.',
        'It misses domain-specific IDs — that is why we add an **EntityRuler** for case-specific patterns.',
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
      title: 'Patterns before statistical NER',
      icon: 'regex',
      points: [
        'Pre-trained model misses `CASE-42`, `TKT-8842`.',
        '`nlp.add_pipe("entity_ruler", before="ner")`',
        'Regex patterns: `CASE-\\d+`, `TKT-\\d+`, `ORD-\\d{4}-\\d+`.',
        'Pipe **order matters** - ruler runs before NER.',
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
        'Optional: `python start.py --real-world`',
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
