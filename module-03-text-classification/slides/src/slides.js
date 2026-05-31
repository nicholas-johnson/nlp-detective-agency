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
      title: 'Module 3 - Text Classification',
      subtitle: 'Sorting tips and witness mood',
      icon: 'tag',
    },
  },
  {
    type: 'welcome',
    content: {
      title: 'Triage the evidence',
      points: [
        'Module 2 turned statements into **numerical fingerprints**.',
        'Now we **label** text and train models to predict those labels.',
        'Two Inkwell tasks: **witness sentiment** (calm vs hostile) and **tip credibility** (credible vs hoax).',
        'scikit-learn **Pipelines** keep vectorisation and classification leak-free.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Learning goals',
      icon: 'target',
      points: [
        'Split data with **`train_test_split`** - stratify, hold out a test set.',
        'Build **`Pipeline`** objects chaining TfidfVectorizer and a classifier.',
        'Compare **MultinomialNB**, **LogisticRegression**, and **LinearSVC**.',
        'Evaluate with **accuracy**, **precision**, **recall**, **F1**, and **confusion matrices**.',
        'Understand **precision vs recall** for filtering tasks like hoax detection.',
      ],
    },
  },

  // ---- Train/test split ----
  {
    type: 'title',
    content: {
      title: 'Train / test split',
      subtitle: 'Hold out data you have not seen',
      icon: 'layers',
    },
  },
  {
    type: 'code',
    content: {
      title: 'Stratified split',
      code: `from sklearn.model_selection import train_test_split

texts = [r["text"] for r in records]
labels = [r["sentiment"] for r in records]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.25,
    random_state=42,
    stratify=labels,   # keep class balance
)`,
      highlights: [
        'Fit on train only - never tune on test',
        'stratify preserves calm/hostile ratio in both splits',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Data leakage warning',
      icon: 'alert-triangle',
      points: [
        'If you **fit** TF-IDF on all data, test vocabulary leaks into training.',
        'A **Pipeline** fit on `X_train` learns vocabulary from train only.',
        'Cross-validation repeats train/test splits - use for model comparison.',
        'The demo uses `random_state=42` everywhere for reproducibility.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Demo - Dataset overview',
      subtitle:
        'python module-03-text-classification/demo/demo.py - option 1',
      icon: 'terminal',
    },
  },

  // ---- Pipelines ----
  {
    type: 'title',
    content: {
      title: 'Pipelines',
      subtitle: 'Vectorise and classify in one step',
      icon: 'layers',
    },
  },
  {
    type: 'code',
    content: {
      title: 'Pipeline pattern',
      code: `from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB()),
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)`,
      highlights: [
        'One object - fit, predict, cross-validate',
        'Test data never influences the vocabulary',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Demo - Sentiment pipeline',
      subtitle:
        'python module-03-text-classification/demo/demo.py - option 2',
      icon: 'terminal',
    },
  },

  // ---- Classifiers ----
  {
    type: 'title',
    content: {
      title: 'Three classifiers',
      subtitle: 'Baselines for text classification',
      icon: 'brain',
    },
  },
  {
    type: 'cards',
    content: {
      title: 'Classifier trade-offs',
      cards: [
        {
          title: 'MultinomialNB',
          icon: 'zap',
          points: [
            'Fast on sparse TF-IDF features',
            'Strong baseline for short text',
            'Assumes feature independence',
          ],
        },
        {
          title: 'LogisticRegression',
          icon: 'scale',
          points: [
            'Linear decision boundary',
            'Probabilistic outputs',
            'Often best on small tabular-text data',
          ],
        },
        {
          title: 'LinearSVC',
          icon: 'shield',
          points: [
            'Maximum-margin linear classifier',
            'Robust with high-dimensional sparse input',
            'Use dual="auto" for wide matrices',
          ],
        },
      ],
    },
  },

  {
    type: 'equation',
    content: {
      title: 'Naive Bayes',
      mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>P</mi><mo>(</mo><mi>c</mi><mo>∣</mo><mi>d</mi><mo>)</mo><mo>∝</mo><mi>P</mi><mo>(</mo><mi>c</mi><mo>)</mo><munder><mo>∏</mo><mi>w</mi></munder><mi>P</mi><mo>(</mo><mi>w</mi><mo>∣</mo><mi>c</mi><mo>)</mo></math>',
      description: "Multiply the prior class probability by each word's likelihood given that class. 'Naive' means we assume words are independent — wrong but effective.",
    },
  },
  {
    type: 'equation',
    content: {
      title: 'Logistic regression — the sigmoid',
      mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>P</mi><mo>(</mo><mi>y</mi><mo>=</mo><mn>1</mn><mo>∣</mo><mi mathvariant="bold">x</mi><mo>)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><mo>(</mo><mi mathvariant="bold">w</mi><mo>·</mo><mi mathvariant="bold">x</mi><mo>+</mo><mi>b</mi><mo>)</mo></mrow></msup></mrow></mfrac></math>',
      description: "The sigmoid squashes any real number into (0, 1) — interpretable as a probability. Positive weights push toward class 1; negative weights push away.",
    },
  },
  {
    type: 'equation',
    content: {
      title: 'SVM margin',
      mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mtext>margin</mtext><mo>=</mo><mfrac><mn>2</mn><mrow><mo>‖</mo><mi mathvariant="bold">w</mi><mo>‖</mo></mrow></mfrac></math>',
      description: "SVM finds the decision boundary that maximises the gap between the two nearest class examples. Smaller weights = wider margin = better generalisation.",
    },
  },
  {
    type: 'title',
    content: {
      title: 'Demo - Hoax pipeline and classifier shootout',
      subtitle:
        'python module-03-text-classification/demo/demo.py - options 3 & 4',
      icon: 'terminal',
    },
  },

  // ---- Sentiment ----
  {
    type: 'title',
    content: {
      title: 'Sentiment triage',
      subtitle: 'Calm or hostile witness statements',
      icon: 'message-square',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Binary sentiment labels',
      icon: 'tag',
      points: [
        '**calm** - factual, cooperative tone (Finch, Hayes)',
        '**hostile** - defensive, accusatory, agitated (Marsh)',
        'Balanced labels (7/7) - accuracy is a fair headline metric',
        'F1 with `pos_label="hostile"` focuses on catching agitation',
      ],
    },
  },

  // ---- Hoax detection ----
  {
    type: 'title',
    content: {
      title: 'Hoax detection',
      subtitle: 'Credible tips vs time-wasters',
      icon: 'filter',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Precision vs recall for filters',
      icon: 'scale',
      points: [
        '**False negative** - hoax marked credible → wasted detective hours',
        '**False positive** - credible tip discarded → missed lead',
        'Filtering tasks often prioritise **recall** on the positive class',
        'Our tips are slightly imbalanced (10 credible / 8 hoax)',
      ],
    },
  },

  // ---- Evaluation ----
  {
    type: 'title',
    content: {
      title: 'Evaluation metrics',
      subtitle: 'Beyond accuracy',
      icon: 'calculator',
    },
  },
  {
    type: 'code',
    content: {
      title: 'Reports and matrices',
      code: `from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cross_val_score,
    f1_score,
    make_scorer,
)

print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds, labels=["credible", "hoax"]))

hoax_f1 = make_scorer(f1_score, pos_label="hoax")
scores = cross_val_score(pipeline, texts, labels, cv=5, scoring=hoax_f1)`,
      highlights: [
        'classification_report - precision, recall, F1 per class',
        'cross_val_score - compare models without a single lucky split',
      ],
    },
  },
  {
    type: 'equation',
    content: {
      title: 'Precision, Recall, F1',
      mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mtable columnalign="right center left" rowspacing="1em"><mtr><mtd><mtext>Precision</mtext></mtd><mtd><mo>=</mo></mtd><mtd><mfrac><mtext>TP</mtext><mrow><mtext>TP</mtext><mo>+</mo><mtext>FP</mtext></mrow></mfrac></mtd></mtr><mtr><mtd><mtext>Recall</mtext></mtd><mtd><mo>=</mo></mtd><mtd><mfrac><mtext>TP</mtext><mrow><mtext>TP</mtext><mo>+</mo><mtext>FN</mtext></mrow></mfrac></mtd></mtr><mtr><mtd><msub><mtext>F</mtext><mn>1</mn></msub></mtd><mtd><mo>=</mo></mtd><mtd><mn>2</mn><mo>·</mo><mfrac><mrow><mtext>Precision</mtext><mo>·</mo><mtext>Recall</mtext></mrow><mrow><mtext>Precision</mtext><mo>+</mo><mtext>Recall</mtext></mrow></mfrac></mtd></mtr></mtable></math>',
      description: "Precision = of predicted positives, how many are correct. Recall = of actual positives, how many were found. F1 is their harmonic mean — it is low if either is low.",
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Reading the confusion matrix',
      icon: 'hash',
      points: [
        'Rows = **actual**, columns = **predicted**.',
        'Diagonal = correct predictions.',
        'Off-diagonal = errors - bottom-left cell = hoax called credible.',
        'Exercise 02 lists those **slipped-through** hoax IDs explicitly.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Demo - Classification report and confusion matrix',
      subtitle:
        'python module-03-text-classification/demo/demo.py - option 5',
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
      title: 'Field rules - Module 3',
      rules: [
        {
          rule: 'Pipeline everything',
          example: 'Never fit TfidfVectorizer on test data separately.',
          icon: 'layers',
        },
        {
          rule: 'Pick metrics for the task',
          example: 'Hoax filters care about false negatives - not just accuracy.',
          icon: 'scale',
        },
        {
          rule: 'Compare with cross-validation',
          example: 'One train/test split can mislead - CV stabilises rankings.',
          icon: 'refresh',
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
        'Run interactively: **`python start.py`**.',
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
        '01 - Sentiment Triage: batch-evaluate calm/hostile on a held-out test set',
        '02 - Hoax Filter: classifier shootout + find hoaxes that slip through',
        '03 - Spam Detector: SMS spam shootout - same pipeline on real messages',
      ],
    },
  },
  {
    type: 'title',
    content: {
      title: 'Module 3 - Complete',
      subtitle: 'Next: Topic Modelling - patterns in the archives',
      icon: 'check-circle',
    },
  },
];
