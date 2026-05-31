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
      title: 'Module 2 - Feature Extraction',
      subtitle: 'Bag-of-Words, n-grams, and TF-IDF',
      icon: 'hash',
    },
  },
  {
    type: 'welcome',
    content: {
      title: 'Dusting for prints',
      points: [
        'The case files are **legible** now - Module 1 cleaned and tokenised them.',
        'Next step: turn text into **numerical signatures** the system can compare.',
        'Bag-of-Words counts words. **TF-IDF** weights what is distinctive.',
        'These classical techniques power search, clustering, and classification.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Learning goals',
      icon: 'target',
      points: [
        'Represent documents as **Bag-of-Words** vectors and understand the document-term matrix.',
        'Capture word order with **n-grams** (unigrams, bigrams, trigrams).',
        'Apply **TF-IDF** weighting to emphasise distinctive terms.',
        'Use **CountVectorizer** and **TfidfVectorizer** in scikit-learn.',
        'Tune **vocabulary size**, **min/max df**, and **n-gram range**.',
      ],
    },
  },

  // ---- Bag-of-Words ----
  {
    type: 'title',
    content: {
      title: 'Bag-of-Words',
      subtitle: 'The word fingerprint',
      icon: 'hash',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Document-term matrix',
      icon: 'hash',
      points: [
        'Ignore grammar, ignore order - **just count**.',
        'Each document becomes a vector of word frequencies.',
        'All documents share one **vocabulary** - the union of all unique terms.',
        'Matrices are **sparse**: most entries are zero.',
        'Makes text comparable with basic maths - dot products, cosine similarity.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'CountVectorizer',
      code: `from sklearn.feature_extraction.text import CountVectorizer

texts = [
    "the dock was empty",
    "the warehouse was empty",
    "the dock was busy",
]
vectorizer = CountVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(texts)

print(matrix.shape)                        # (3, 3)
print(vectorizer.get_feature_names_out())    # ['busy' 'dock' 'empty' ...]`,
      highlights: [
        'stop_words="english" removes the, was, and',
        'fit_transform learns vocabulary and converts in one step',
      ],
    },
  },
  {
    type: 'title',
    content: {
      title: 'Demo - BoW matrix and top terms',
      subtitle:
        'python module-02-feature-extraction/demo/demo.py - options 2 & 3',
      icon: 'terminal',
    },
  },

  // ---- N-grams ----
  {
    type: 'title',
    content: {
      title: 'N-grams',
      subtitle: 'Catching phrases, not just words',
      icon: 'layers',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Unigrams, bigrams, trigrams',
      icon: 'layers',
      points: [
        '**Unigram** - single words: `dock`, `warehouse`.',
        '**Bigram** - word pairs: `near dock`, `dock workers`.',
        '**Trigram** - three words: `near the dock`.',
        '`ngram_range=(1, 2)` includes unigrams **and** bigrams.',
        'Higher n captures context but **explodes vocabulary size**.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'N-gram vocabulary',
      code: `vectorizer = CountVectorizer(
    stop_words="english",
    ngram_range=(1, 2),   # unigrams + bigrams
)
matrix = vectorizer.fit_transform(texts)
print(len(vectorizer.get_feature_names_out()))
# More features than unigrams alone`,
      highlights: [
        '(1, 1) = unigrams only',
        '(1, 3) = unigrams + bigrams + trigrams',
      ],
    },
  },
  {
    type: 'title',
    content: {
      title: 'Demo - N-gram vocabulary sizes',
      subtitle:
        'python module-02-feature-extraction/demo/demo.py - option 5',
      icon: 'terminal',
    },
  },

  // ---- TF-IDF ----
  {
    type: 'title',
    content: {
      title: 'TF-IDF',
      subtitle: 'Weighting what matters',
      icon: 'scale',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Term frequency x inverse document frequency',
      icon: 'scale',
      points: [
        '**TF** - how often a term appears in *this* document.',
        '**IDF** - penalises terms that appear in *many* documents.',
        'Common words (`said`, `the`) get low scores everywhere.',
        'Distinctive words (`warehouse`, `reeves`) score high in the right document.',
        'Better than raw counts for finding what makes each document unique.',
      ],
    },
  },
  {
    type: 'equation',
    content: {
      title: 'TF-IDF',
      mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mtext>tf-idf</mtext><mo>(</mo><mi>t</mi><mo>,</mo><mi>d</mi><mo>)</mo><mo>=</mo><mtext>tf</mtext><mo>(</mo><mi>t</mi><mo>,</mo><mi>d</mi><mo>)</mo><mo>×</mo><mtext>idf</mtext><mo>(</mo><mi>t</mi><mo>)</mo><mspace width="2em"/><mtext>where</mtext><mspace width="1em"/><mtext>idf</mtext><mo>(</mo><mi>t</mi><mo>)</mo><mo>=</mo><mi>log</mi><mfrac><mi>N</mi><mrow><mtext>df</mtext><mo>(</mo><mi>t</mi><mo>)</mo></mrow></mfrac></math>',
      description: "TF counts how often a term appears in this document. IDF penalises terms that appear in many documents. The product highlights words that are frequent here but rare overall.",
    },
  },
  {
    type: 'equation',
    content: {
      title: 'scikit-learn IDF (smoothed)',
      mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mtext>idf</mtext><mo>(</mo><mi>t</mi><mo>)</mo><mo>=</mo><mi>log</mi><mfrac><mrow><mn>1</mn><mo>+</mo><mi>N</mi></mrow><mrow><mn>1</mn><mo>+</mo><mtext>df</mtext><mo>(</mo><mi>t</mi><mo>)</mo></mrow></mfrac><mo>+</mo><mn>1</mn></math>',
      description: "scikit-learn adds 1 to numerator and denominator to avoid division by zero, and adds 1 to the result so no term gets zero weight.",
    },
  },
  {
    type: 'code',
    content: {
      title: 'TfidfVectorizer',
      code: `from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(texts)

row = matrix[0].toarray().flatten()
names = vectorizer.get_feature_names_out()
top = row.argsort()[-3:][::-1]
print([names[i] for i in top])  # highest TF-IDF terms`,
      highlights: [
        'Same interface as CountVectorizer',
        'Values are floats (weights), not integer counts',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'Cosine similarity',
      code: `from sklearn.metrics.pairwise import cosine_similarity

sims = cosine_similarity(matrix)
# sims[i][j] = similarity between doc i and j
# 0 = no overlap, 1 = identical

score = sims[0][1]  # how similar are witnesses 0 and 1?`,
      highlights: [
        'Find witness statements telling similar stories',
        'Skip the diagonal - a doc compared to itself is always 1.0',
      ],
    },
  },
  {
    type: 'equation',
    content: {
      title: 'Cosine similarity',
      mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>cos</mi><mo>(</mo><mi>θ</mi><mo>)</mo><mo>=</mo><mfrac><mrow><mi mathvariant="bold">a</mi><mo>·</mo><mi mathvariant="bold">b</mi></mrow><mrow><mo>‖</mo><mi mathvariant="bold">a</mi><mo>‖</mo><mspace width="0.2em"/><mo>‖</mo><mi mathvariant="bold">b</mi><mo>‖</mo></mrow></mfrac></math>',
      description: "Measures direction, not magnitude. 1 = identical direction, 0 = no overlap, −1 = opposite. This is how we compare document fingerprints.",
    },
  },
  {
    type: 'title',
    content: {
      title: 'Demo - TF-IDF and similarity',
      subtitle:
        'python module-02-feature-extraction/demo/demo.py - options 4 & 6',
      icon: 'terminal',
    },
  },

  // ---- Tuning ----
  {
    type: 'title',
    content: {
      title: 'Tuning vectorisers',
      subtitle: 'Controlling vocabulary size and quality',
      icon: 'sliders',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Key parameters',
      icon: 'sliders',
      points: [
        '`min_df=2` - ignore terms appearing in fewer than 2 documents.',
        '`max_df=0.9` - ignore terms in more than 90% of documents.',
        '`max_features=5000` - cap vocabulary at the top N terms.',
        '`ngram_range` - control phrase capture vs vocabulary explosion.',
        '**Fit on training data only** - transform test data with the same vocabulary.',
      ],
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
      title: 'Field rules - Module 2',
      rules: [
        {
          rule: 'Fit on training data only',
          example: 'Leaking test vocabulary into fit() inflates your metrics.',
          icon: 'shield',
        },
        {
          rule: 'Sparse matrices are normal',
          example: 'Do not call .toarray() on large corpora - stay sparse.',
          icon: 'database',
        },
        {
          rule: 'Inspect your vocabulary',
          example: 'Surprising tokens often reveal preprocessing gaps.',
          icon: 'search',
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
        'Install: **`pip install -e ".[nlp,dev]"`** (includes scikit-learn).',
        'Run interactively: **`python start.py CASE-42`**.',
        'Run tests from the exercise folder: **`pytest test_start.py -v`**.',
        'Solutions in **`solution.py`** - try the exercise first!',
      ],
    },
  },
  {
    type: 'welcome',
    content: {
      title: 'Exercises',
      points: [
        '01 - Document Fingerprints: BoW fingerprint card for every witness statement in a case',
        '02 - Matching Prints: TF-IDF cosine similarity to find the most similar witness pair',
        '03 - Article Matcher: 20 Newsgroups - validate TF-IDF similarity against real categories',
      ],
    },
  },
  {
    type: 'title',
    content: {
      title: 'Module 2 - Complete',
      subtitle: 'Next: Text Classification - sorting the evidence',
      icon: 'check-circle',
    },
  },
];
