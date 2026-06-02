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
      title: 'Module 1 - Text Preprocessing',
      subtitle: 'Tokenise, clean, and prepare case files for analysis',
      icon: 'file-text',
    },
  },
  {
    type: 'welcome',
    content: {
      title: 'The evidence prep room',
      points: [
        'You have joined **Inkwell Investigations** - a detective agency drowning in unsorted case files.',
        'Witness statements, tip-off letters, and clippings arrive messy: ALL CAPS, shorthand, redactions.',
        'Before any analysis begins, every document needs to be **cleaned, normalised, and tokenised**.',
        'This module is your evidence prep room: turn chaotic prose into machine-readable text.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Learning goals',
      icon: 'target',
      points: [
        '**Tokenise** text at word and sentence level using NLTK (and compare with spaCy).',
        'Apply **case folding**, **normalisation**, and **regex cleaning** to raw documents.',
        'Remove **stopwords** and handle punctuation, numbers, and unicode artefacts.',
        'Compare **stemming** vs **lemmatisation** and know when to use each.',
        'Build a reusable **preprocessing pipeline** that chains steps together.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'Setup - one-time NLTK download',
      code: `pip install -e ".[nlp,dev]"

python -c "
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
"`,
      highlights: [
        'punkt_tab - sentence tokenisation',
        'stopwords + wordnet - filtering and lemmatisation',
      ],
    },
  },

  // ---- Section: Tokenisation ----
  {
    type: 'title',
    content: {
      title: 'Tokenisation',
      subtitle: 'Breaking text into sentences and words',
      icon: 'scissors',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Why tokenise?',
      icon: 'scissors',
      points: [
        'A witness statement is a wall of prose - tokenisation splits it into examinable pieces.',
        '**Sentence tokenisation** finds sentence boundaries (`.`, `!`, `?`).',
        'Trickier than it looks: *"Dr. Reeves arrived at 4 p.m."* is **one** sentence, not four.',
        '**Word tokenisation** splits into tokens - words and punctuation as separate items.',
        'Always **normalise first**, then tokenise - or `DOCKS` and `docks` count as different tokens.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'Sentence and word tokenisation',
      code: `from nltk.tokenize import sent_tokenize, word_tokenize

text = (
    "I saw HIM near the DOCKS on Tuesday!!! "
    "He was with a man called Reeves."
)

sent_tokenize(text)
# ['I saw HIM near the DOCKS on Tuesday!!!',
#  'He was with a man called Reeves.']

word_tokenize("Reeves-or so I thought.")
# ['Reeves', '-', 'or', 'so', 'I', 'thought', '.']`,
      highlights: [
        'sent_tokenize - sentence boundaries',
        'word_tokenize - words + punctuation as separate tokens',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'NLTK vs spaCy',
      icon: 'git-branch',
      points: [
        '**NLTK** - learning, quick scripts, classical NLP. Output: lists of strings.',
        '**spaCy** - production pipelines, NER, POS tagging. Output: rich `Doc` objects.',
        'NLTK setup: `nltk.download("punkt_tab")` - no model download.',
        'spaCy setup: `python -m spacy download en_core_web_sm` - larger install.',
        'This module uses **NLTK** in exercises. You will meet spaCy again in Module 6.',
      ],
    },
  },
  {
    type: 'title',
    content: {
      title: 'Demo - Tokenisation',
      subtitle:
        'python module-01-text-preprocessing/demo/demo.py - menu options 2 & 3',
      icon: 'terminal',
    },
  },

  // ---- Section: Cleaning ----
  {
    type: 'title',
    content: {
      title: 'Cleaning and normalisation',
      subtitle: 'Scrubbing the grime from case files',
      icon: 'eraser',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'What needs cleaning?',
      icon: 'eraser',
      points: [
        '**Case folding** - lowercase so `DOCKS` and `docks` match.',
        '**Redactions** - `[REDACTED]` blocks carry no semantic value; strip them.',
        '**Case references** - `CASE-42` scattered through prose; remove for topic analysis.',
        '**Whitespace** - collapse runs of spaces, tabs, and newlines.',
        '**Unicode** - normalise curly quotes and em-dashes with `unicodedata.normalize("NFKC", text)`.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'normalize_text',
      code: `import re

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\\[redacted\\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"case-\\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

# "I saw HIM near the DOCKS!!!  CASE-42"
# → "i saw him near the docks!!!"`,
      highlights: [
        'Apply cleaning BEFORE tokenisation',
        'Always keep the original raw text for audit trails',
      ],
    },
  },
  {
    type: 'title',
    content: {
      title: 'Demo - Cleaning',
      subtitle:
        'python module-01-text-preprocessing/demo/demo.py - menu option 4',
      icon: 'terminal',
    },
  },

  // ---- Section: Stopwords, stemming, lemmatisation ----
  {
    type: 'title',
    content: {
      title: 'Stopwords, stemming, lemmatisation',
      subtitle: 'Trimming the noise from witness statements',
      icon: 'filter',
    },
  },
  {
    type: 'equation',
    content: {
      title: "Zipf's law",
      mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>f</mi><mo>(</mo><mi>r</mi><mo>)</mo><mo>∝</mo><mfrac><mn>1</mn><msup><mi>r</mi><mi>s</mi></msup></mfrac><mo>,</mo><mspace width="1em"/><mi>s</mi><mo>≈</mo><mn>1</mn></math>',
      points: [
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>f</mi><mo>(</mo><mi>r</mi><mo>)</mo></math>',
          text: '— frequency of the word at rank *r*.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>r</mi></math>',
          text: '— rank of the word when sorted by frequency (1 = most common).',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>s</mi><mo>≈</mo><mn>1</mn></math>',
          text: '— exponent close to 1 for natural language, making frequency inversely proportional to rank.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mfrac><mn>1</mn><msup><mi>r</mi><mi>s</mi></msup></mfrac></math>',
          text: '— a few top-ranked words dominate all counts; the long tail has many rare words.',
        },
      ],
      credit: 'Zipf, 1949',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Stopwords',
      icon: 'filter',
      points: [
        '*"the"*, *"a"*, *"is"* tell you nothing about a suspect.',
        'English stopword lists contain ~170 high-frequency function words.',
        'Filter with `token.isalpha()` - drop punctuation tokens too.',
        'Stopword removal is task-dependent: keep them for sentiment if negation matters.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'Removing stopwords',
      code: `from nltk.corpus import stopwords

stops = set(stopwords.words("english"))
tokens = word_tokenize(normalize_text(text))

filtered = [
    t for t in tokens
    if t.isalpha() and t not in stops
]
# "the docks were empty" → ["docks", "empty"]`,
      highlights: [
        'Load stopwords once at module level',
        'Keep only alphabetic tokens',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Stemming',
      icon: 'scissors',
      points: [
        '**Stemming** strips suffixes with a cascade of rules - no dictionary lookup.',
        'Porter (1980): five phases of suffix rules (`-s`, `-ed`, `-ing`, `-ational`, …).',
        'Goal: collapse inflected forms so they match in search and bag-of-words.',
        '`investigating`, `investigated`, `investigation` → all become **`investig`**.',
        'Fast, but stems are not always real words (`studies` → `studi`).',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'PorterStemmer',
      code: `from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

[stemmer.stem(w) for w in [
    "investigating",
    "investigated",
    "investigation",
    "studies",
]]
# ['investig', 'investig', 'investig', 'studi']

# SnowballStemmer("english") is Porter's improved successor`,
      highlights: [
        'All investigate-forms collapse to the same stem',
        'Good for search indexing and high-volume bag-of-words',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Lemmatisation',
      icon: 'book-open',
      points: [
        'A **lemma** is the dictionary headword: `go`, `goes`, `went`, `gone` → **`go`**.',
        '**Lemmatisation** maps each token to a valid dictionary form using **WordNet**.',
        'Strip suffixes, then **validate against the lexicon** - unlike stemming, which stops after rules.',
        '**POS matters**: `saw` (verb) → `see`, but `saw` (noun) → `saw`.',
        'Default is noun - pass `pos="v"` for verbs or `investigating` stays unchanged.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'WordNetLemmatizer',
      code: `from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

lemmatizer.lemmatize("docks")                 # 'dock'
lemmatizer.lemmatize("investigating")         # 'investigating' (noun default!)
lemmatizer.lemmatize("investigating", pos="v")  # 'investigate'
lemmatizer.lemmatize("saw", pos="v")          # 'see'
lemmatizer.lemmatize("saw", pos="n")          # 'saw'`,
      highlights: [
        'pos="v" for verbs, "a" adjectives, "r" adverbs, "n" nouns',
        'Module 6: use spaCy POS tags to lemmatise automatically',
      ],
    },
  },
  {
    type: 'cards',
    content: {
      title: 'Stemming vs lemmatisation',
      cards: [
        {
          heading: 'Stemming (Porter)',
          points: [
            '**Rule-based** - chop suffixes with heuristics.',
            'Fast; good for search indexing and bag-of-words.',
            '`investigating` → `investig`, `investigated` → `investig`.',
            'Can produce non-words; no vocabulary lookup.',
          ],
        },
        {
          heading: 'Lemmatisation (WordNet)',
          points: [
            '**Dictionary-based** - reduce to root form using vocabulary + POS.',
            'Slower but more accurate; produces real words.',
            '`docks` → `dock`, `investigating` (verb) → `investigate`.',
            'Hint POS with `pos="v"` for verbs, `pos="a"` for adjectives.',
          ],
        },
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'Stem vs lemma side by side',
      code: `from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ["investigating", "investigated", "docks"]
stems   = [stemmer.stem(w) for w in words]
lemmas  = [lemmatizer.lemmatize(w) for w in words]

# stems:  ['investig', 'investig', 'dock']
# lemmas: ['investigating', 'investigated', 'dock']`,
      highlights: [
        'Stemming collapses all three investigate-forms to "investig"',
        'Lemmatisation needs POS hint for verbs - default is noun',
      ],
    },
  },
  {
    type: 'title',
    content: {
      title: 'Demo - Stopwords and stem vs lemma',
      subtitle:
        'python module-01-text-preprocessing/demo/demo.py - menu options 5 & 6',
      icon: 'terminal',
    },
  },

  // ---- Section: Pipeline ----
  {
    type: 'title',
    content: {
      title: 'The preprocessing pipeline',
      subtitle: 'Chaining steps into a reusable assembly line',
      icon: 'layers',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Pure functions, chained',
      icon: 'layers',
      points: [
        'One-off cleaning scripts do not scale - chain steps into one function.',
        'Each step is a **pure function**: text in, transformed output out.',
        'Easy to test each step independently (`normalize_text`, `remove_stopwords`, etc.).',
        'For batch work: map across documents, aggregate with `collections.Counter`.',
        'Exercise 01 audits statements; Exercise 02 builds a **case briefing** from the pipeline.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'preprocess_statement',
      code: `def preprocess_statement(text: str) -> list[str]:
    text = normalize_text(text)
    tokens = word_tokenize(text)
    tokens = remove_stopwords(tokens)
    return lemmatize_tokens(tokens)

# Raw witness statement → list of clean lemma tokens
# ready for counting, classification, or topic modelling`,
      highlights: [
        'normalize → tokenise → stopwords → lemmatise',
        'Same pipeline in demo option 7 and Exercise 02',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'Case briefing with Counter',
      code: `from collections import Counter

def case_briefing(statements, case_id, top_n=10):
    texts = [
        s["raw_text"]
        for s in statements
        if s["case_id"] == case_id
    ]
    counter = Counter()
    for text in texts:
        for token in preprocess_statement(text):
            if len(token) >= 3:
                counter[token] += 1
    return counter.most_common(top_n)

# CASE-42 → [('dock', 4), ('warehouse', 3), ...]`,
      highlights: [
        'Demo option 8: briefing across the full archive',
        'Exercise 02: briefing for a single case_id only',
      ],
    },
  },
  {
    type: 'title',
    content: {
      title: 'Demo - Full pipeline and briefing',
      subtitle:
        'python module-01-text-preprocessing/demo/demo.py - menu options 7 & 8',
      icon: 'terminal',
    },
  },

  // ---- Wrap-up ----
  {
    type: 'title',
    content: {
      title: 'Putting it all together',
      subtitle: 'Field rules and exercises',
      icon: 'check-square',
    },
  },
  {
    type: 'rules',
    content: {
      title: 'Field rules - Module 1',
      rules: [
        {
          rule: 'Never leak test data into preprocessing',
          example:
            'Fit vectorisers and statistics on training data only - transform test separately.',
          icon: 'shield',
        },
        {
          rule: 'Stemming is fast; lemmatisation is accurate',
          example: 'Pick based on downstream task - briefings want lemmas, search wants stems.',
          icon: 'scale',
        },
        {
          rule: 'Keep the original text',
          example:
            'Always store raw text alongside cleaned versions for audit trails.',
          icon: 'file-text',
        },
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Getting started with exercises',
      icon: 'settings',
      points: [
        'Install NLP extras: **`pip install -e ".[nlp,dev]"`**.',
        'Each exercise has **`start.py`** (your work) and **`test_start.py`** (pytest).',
        'Run interactively: **`python start.py`** - see your output before running tests.',
        'Run tests from the exercise folder: **`pytest test_start.py -v`**.',
        'Solutions are in **`solution.py`** - try the exercise first!',
      ],
    },
  },
  {
    type: 'welcome',
    content: {
      title: 'Exercises',
      points: [
        '01 - Statement Audit: batch-audit every witness statement; flag long or complex statements for review',
        '02 - Case Briefing: build the preprocessing pipeline and generate a word-frequency briefing for one case',
        '03 - Review Scanner: movie reviews - compare positive vs negative vocabulary after preprocessing',
      ],
    },
  },
  {
    type: 'title',
    content: {
      title: 'Module 1 - Complete',
      subtitle: 'Next: Feature Extraction - dusting for prints',
      icon: 'check-circle',
    },
  },
];
