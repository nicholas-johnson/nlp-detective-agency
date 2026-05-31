export const slides = [
  {
    type: 'title',
    content: {
      title: 'Python NLP',
      subtitle: 'Inkwell Investigations — Nicholas Johnson',
      icon: 'search',
    },
  },
  {
    type: 'title',
    content: {
      title: 'Module 0 — Python for Text Analysis',
      subtitle: 'The analyst\'s toolkit',
      icon: 'file-text',
    },
  },
  {
    type: 'welcome',
    content: {
      title: 'Welcome to Inkwell Investigations',
      points: [
        'Decades of case files: witness statements, tip-off letters, cold cases.',
        'Before any NLP, you need **Python fluency with text**.',
        'This module covers the fundamentals every later module depends on.',
        'Strings, regex, file I/O, collections, and the NLP library landscape.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Learning goals',
      icon: 'target',
      points: [
        'Work fluently with Python **strings**: slicing, methods, f-strings.',
        'Use **regular expressions** to search and extract patterns.',
        'Read files with **pathlib** and parse **JSON** data.',
        'Count and group with **Counter**, **defaultdict**, and comprehensions.',
        'Write **type-hinted** function signatures.',
        'Run scripts from the **command line**.',
      ],
    },
  },

  // ---- Strings ----
  {
    type: 'title',
    content: {
      title: 'Strings and Slicing',
      subtitle: 'Reading the fine print',
      icon: 'scissors',
    },
  },
  {
    type: 'code',
    content: {
      title: 'Indexing and slicing',
      language: 'python',
      code: `statement = "I saw Reeves near the docks on Tuesday"

statement[0]       # 'I'
statement[-1]      # 'y'
statement[6:12]    # 'Reeves'
statement[:5]      # 'I saw'

# Slices: [start:stop:step]
# stop is EXCLUSIVE — len(s[a:b]) == b - a`,
    },
  },
  {
    type: 'code',
    content: {
      title: 'Essential string methods',
      language: 'python',
      code: `raw = "  I saw HIM near the DOCKS on Tuesday!!!  "

raw.strip()              # remove whitespace
raw.lower()              # all lowercase
raw.upper()              # ALL CAPS
raw.replace("!!!", ".")  # swap punctuation
raw.split()              # whitespace tokens → list
" ".join(words)          # rejoin with single space

# Chain them: raw.strip().lower().replace("!!!", "")`,
    },
  },
  {
    type: 'code',
    content: {
      title: 'f-strings',
      language: 'python',
      code: `witness = "Margaret Hayes"
case_id = "CASE-42"
word_total = 30

print(f"Witness: {witness} | Case: {case_id} | Words: {word_total}")
# Witness: Margaret Hayes | Case: CASE-42 | Words: 30

# Expressions work too
print(f"Uppercase: {witness.upper()}")`,
    },
  },

  // ---- Regex ----
  {
    type: 'title',
    content: {
      title: 'Regular Expressions',
      subtitle: 'Pattern matching at scale',
      icon: 'search',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Core re functions',
      icon: 'list',
      points: [
        '`re.search(pattern, text)` — first match anywhere.',
        '`re.findall(pattern, text)` — all non-overlapping matches.',
        '`re.sub(pattern, repl, text)` — replace every match.',
        '`re.compile(pattern)` — pre-compile for repeated use.',
        'Always use **raw strings**: `r"\\d+"` not `"\\\\d+"`.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'Practical patterns',
      language: 'python',
      code: `import re

text = "[REDACTED] told me. CASE-42 is closed. Date: 1947-03-12."

re.findall(r"CASE-\\d+", text)         # ['CASE-42']
re.findall(r"\\[REDACTED\\]", text)     # ['[REDACTED]']
re.findall(r"\\d{4}-\\d{2}-\\d{2}", text) # ['1947-03-12']
re.sub(r"[^a-zA-Z\\s]", "", text)       # strip punctuation

# Flags
re.findall(r"reeves", text, re.IGNORECASE)`,
    },
  },
  {
    type: 'code',
    content: {
      title: 'Capture groups',
      language: 'python',
      code: `import re

m = re.search(r"(CASE)-(\\d+)", "CASE-42 is closed")
m.group(0)  # 'CASE-42'  (full match)
m.group(1)  # 'CASE'
m.group(2)  # '42'

# findall with groups returns tuples
re.findall(r"(CASE)-(\\d+)", "CASE-42 and CASE-107")
# [('CASE', '42'), ('CASE', '107')]`,
    },
  },

  // ---- File I/O ----
  {
    type: 'title',
    content: {
      title: 'File I/O and Encoding',
      subtitle: 'Opening the archives',
      icon: 'folder-open',
    },
  },
  {
    type: 'code',
    content: {
      title: 'pathlib + JSON — the course pattern',
      language: 'python',
      code: `import json
from pathlib import Path

# Locate data relative to the script
DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data"
path = DATA_DIR / "inkwell" / "statements.json"

# Load
records = json.loads(path.read_text())

# Key Path methods
path.exists()      # True / False
path.name          # 'statements.json'
path.stem          # 'statements'
path.suffix        # '.json'`,
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Encoding matters',
      icon: 'alert-triangle',
      points: [
        'UTF-8 is the modern default — handles accented chars, em-dashes, etc.',
        'Always specify encoding: `open(f, encoding="utf-8")`.',
        'Python 3 strings are Unicode internally — once loaded correctly, everything works.',
        '`pathlib.read_text()` defaults to UTF-8 — another reason to prefer it.',
      ],
    },
  },

  // ---- Collections ----
  {
    type: 'title',
    content: {
      title: 'Collections and Comprehensions',
      subtitle: 'Counting the evidence',
      icon: 'bar-chart',
    },
  },
  {
    type: 'code',
    content: {
      title: 'Comprehensions — filter and transform',
      language: 'python',
      code: `records = json.loads(path.read_text())

# List: extract fields
texts = [r["raw_text"] for r in records]

# With filter
case_42 = [r for r in records if r["case_id"] == "CASE-42"]

# Dict: lookup map
by_id = {r["id"]: r for r in records}

# Set: unique values
witnesses = {r["witness"] for r in records}

# Nested: flatten
all_words = [w for r in records for w in r["raw_text"].split()]`,
    },
  },
  {
    type: 'code',
    content: {
      title: 'Counter and defaultdict',
      language: 'python',
      code: `from collections import Counter, defaultdict

# Counter — word frequencies
words = "the docks the warehouse the pier seven".split()
freq = Counter(words)
freq.most_common(3)   # [('the', 3), ('docks', 1), ...]

# Counter arithmetic
freq_a + freq_b       # merge counts

# defaultdict — group records
by_case = defaultdict(list)
for r in records:
    by_case[r["case_id"]].append(r)`,
    },
  },

  // ---- Type hints ----
  {
    type: 'title',
    content: {
      title: 'Type Hints',
      subtitle: 'Self-documenting function signatures',
      icon: 'tag',
    },
  },
  {
    type: 'code',
    content: {
      title: 'Annotation style used in this course',
      language: 'python',
      code: `def preprocess(text: str) -> list[str]:
    return text.lower().split()

def filter_by_case(records: list[dict], case_id: str) -> list[dict]:
    return [r for r in records if r["case_id"] == case_id]

def find_witness(records: list[dict], name: str) -> dict | None:
    for r in records:
        if r["witness"] == name:
            return r
    return None

# Common types: str, int, float, bool
# Containers: list[str], dict[str, int], set[str]
# Optional: str | None  (Python 3.10+)`,
    },
  },

  // ---- CLI ----
  {
    type: 'code',
    content: {
      title: 'CLI basics',
      language: 'python',
      code: `import sys

# Simple: sys.argv
if len(sys.argv) > 1:
    case_id = sys.argv[1]

# Rich: argparse (used from Module 6 onward)
import argparse

parser = argparse.ArgumentParser(description="Analyse statements")
parser.add_argument("case_id", help="e.g. CASE-42")
parser.add_argument("--limit", "-n", type=int, default=10)
args = parser.parse_args()

# if __name__ == "__main__": guard
# lets a file work as both a script and an importable module`,
    },
  },

  // ---- NLP toolkit ----
  {
    type: 'title',
    content: {
      title: 'The NLP Toolkit',
      subtitle: 'Knowing your instruments',
      icon: 'tool',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Libraries we use in this course',
      icon: 'package',
      points: [
        '**NLTK** — classical NLP: tokenisation, stemming, stopwords, WordNet. (Module 1)',
        '**scikit-learn** — ML pipelines: vectorisers, classifiers, metrics. (Module 2)',
        '**gensim** — topic modelling (LDA) and Word2Vec training. (Module 4)',
        '**spaCy** — production pipeline: POS, NER, dependency parsing. (Module 6)',
        '**Hugging Face Transformers** — pre-trained models, pipelines, fine-tuning. (Module 7)',
        'All installed via `pip install -e ".[nlp,dev]"`.',
      ],
    },
  },

  // ---- Exercises ----
  {
    type: 'standard',
    content: {
      title: 'Exercises',
      icon: 'edit',
      points: [
        '**01 — String Sleuthing**: slice, search, normalise witness strings.',
        '**02 — Regex Extraction**: extract dates, case IDs, redacted markers.',
        '**03 — Corpus Loader**: pathlib + JSON + Counter + comprehensions.',
        'Run `pytest module-00-python-for-text/` to check your work.',
        'Solutions are in `solution.py` — try the exercise first!',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Field rules',
      icon: 'shield',
      points: [
        '**Always specify encoding.** Default assumptions break on real text.',
        '**Regex is powerful but fragile.** Test on real samples, not toy data.',
        '**Use the right data structure.** Lists for sequences, dicts for lookups, Counter for frequencies.',
        '**Type-hint your functions.** Costs seconds, saves hours.',
        '**Comprehensions over loops** — when the logic fits a single step.',
      ],
    },
  },
];
