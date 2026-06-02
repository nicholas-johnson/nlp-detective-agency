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
        'Meet **three classifiers one at a time** — Naive Bayes, Logistic Regression, and Linear SVM.',
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
        'Explain what a **model** is in supervised text classification.',
        'Split data with **`train_test_split`** - stratify, hold out a test set.',
        'Build **`Pipeline`** objects chaining TfidfVectorizer and a classifier.',
        'Compare **MultinomialNB**, **LogisticRegression**, and **LinearSVC**.',
        'Evaluate with **accuracy**, **precision**, **recall**, **F1**, and **confusion matrices**.',
        'Understand **precision vs recall** for filtering tasks like hoax detection.',
      ],
    },
  },

  // ---- What is a model? ----
  {
    type: 'title',
    content: {
      title: 'What is a model?',
      subtitle: 'Supervised learning',
      icon: 'brain',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Supervised text classification',
      icon: 'book-open',
      points: [
        'A **supervised** model learns from **labelled pairs** — each text comes with a known category.',
        'Training (`fit`) discovers which word patterns predict which label.',
        'Prediction (`predict`) applies those patterns to text the model has never seen.',
        'The **Pipeline** (vectoriser + classifier) *is* the model object — one call to `fit`, one call to `predict`.',
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
      subtitle: 'python module-03-text-classification/demo/demo.py - option 1',
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
      subtitle: 'python module-03-text-classification/demo/demo.py - option 2',
      icon: 'terminal',
    },
  },

  // ---- Bayes theorem ----
  {
    type: 'title',
    content: {
      title: "Bayes' theorem",
      subtitle: 'Updating beliefs with evidence',
      icon: 'bar-chart',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Intuition — tips and evidence',
      icon: 'lightbulb',
      points: [
        'Classification question: *given this tip text, how probable is each label?*',
        'Start with a **prior belief** — most tips are credible (base rate from training data).',
        'Then look at the **evidence** — words like "wire $500" or "demand payment" are strong signals for hoax.',
        "Bayes' theorem combines prior belief with evidence to produce an **updated probability**.",
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Three quantities',
      icon: 'list',
      points: [
        '**Prior** P(c) — how common is class c in training data? (base rate of hoaxes vs credible tips)',
        '**Likelihood** P(d | c) — how typical is this document if class c were true?',
        '**Posterior** P(c | d) — what we want: class probability *after* reading the document.',
        'The denominator P(d) is the same for all classes — when comparing, we work with proportions.',
      ],
    },
  },
  {
    type: 'equation',
    content: {
      title: "Bayes' theorem",
      mathml:
        '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>P</mi><mo>(</mo><mi>c</mi><mo>∣</mo><mi>d</mi><mo>)</mo><mo>=</mo><mfrac><mrow><mi>P</mi><mo>(</mo><mi>d</mi><mo>∣</mo><mi>c</mi><mo>)</mo><mspace width="0.2em"/><mi>P</mi><mo>(</mo><mi>c</mi><mo>)</mo></mrow><mrow><mi>P</mi><mo>(</mo><mi>d</mi><mo>)</mo></mrow></mfrac></math>',
      points: [
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>P</mi><mo>(</mo><mi>c</mi><mo>∣</mo><mi>d</mi><mo>)</mo></math>',
          text: '— posterior: probability of class *c* given document *d* (what we want).',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>P</mi><mo>(</mo><mi>d</mi><mo>∣</mo><mi>c</mi><mo>)</mo></math>',
          text: '— likelihood: how well document *d* fits class *c*.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>P</mi><mo>(</mo><mi>c</mi><mo>)</mo></math>',
          text: '— prior: base-rate probability of class *c* before seeing the document.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>P</mi><mo>(</mo><mi>d</mi><mo>)</mo></math>',
          text: '— evidence: probability of the document (constant across classes, so often ignored).',
        },
      ],
    },
  },

  // ---- MultinomialNB ----
  {
    type: 'title',
    content: {
      title: 'Multinomial Naive Bayes',
      subtitle: 'From theorem to text classifier',
      icon: 'zap',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'From Bayes to Naive Bayes',
      icon: 'file-text',
      points: [
        'For text: estimate P(w | c) from word counts in training documents of class c.',
        'The **naive** assumption treats every word as independent given the class.',
        'This lets us replace the full document likelihood with a **product over individual words**.',
        '**Laplace smoothing** (α = 1) prevents unseen words from zeroing out the posterior.',
      ],
    },
  },
  {
    type: 'equation',
    content: {
      title: 'Naive Bayes — word independence',
      mathml:
        '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>P</mi><mo>(</mo><mi>c</mi><mo>∣</mo><mi>d</mi><mo>)</mo><mo>∝</mo><mi>P</mi><mo>(</mo><mi>c</mi><mo>)</mo><munder><mo>∏</mo><mi>w</mi></munder><mi>P</mi><mo>(</mo><mi>w</mi><mo>∣</mo><mi>c</mi><mo>)</mo></math>',
      points: [
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>P</mi><mo>(</mo><mi>c</mi><mo>∣</mo><mi>d</mi><mo>)</mo></math>',
          text: '— posterior probability of class *c* given the document.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mo>∝</mo></math>',
          text: '— proportional to (we drop P(d) since it is the same for all classes).',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>P</mi><mo>(</mo><mi>c</mi><mo>)</mo></math>',
          text: '— prior: fraction of training documents belonging to class *c*.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><munder><mo>∏</mo><mi>w</mi></munder><mi>P</mi><mo>(</mo><mi>w</mi><mo>∣</mo><mi>c</mi><mo>)</mo></math>',
          text: '— product of each word\'s likelihood under class *c* (the "naive" independence assumption).',
        },
      ],
    },
  },
  {
    type: 'equation',
    content: {
      title: 'Worked example — is this message spam?',
      mathml:
        '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>P</mi><mo>(</mo><mtext>spam</mtext><mo>∣</mo><mi>d</mi><mo>)</mo><mo>∝</mo><mi>P</mi><mo>(</mo><mtext>spam</mtext><mo>)</mo><mo>×</mo><mi>P</mi><mo>(</mo><mtext>free</mtext><mo>∣</mo><mtext>spam</mtext><mo>)</mo><mo>×</mo><mi>P</mi><mo>(</mo><mtext>money</mtext><mo>∣</mo><mtext>spam</mtext><mo>)</mo><mo>×</mo><mi>P</mi><mo>(</mo><mtext>offer</mtext><mo>∣</mo><mtext>spam</mtext><mo>)</mo></math>',
      points: [
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>P</mi><mo>(</mo><mtext>spam</mtext><mo>)</mo></math>',
          text: '— prior: what fraction of all training messages were spam.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>P</mi><mo>(</mo><mtext>free</mtext><mo>∣</mo><mtext>spam</mtext><mo>)</mo></math>',
          text: '— likelihood of the word "free" appearing in spam messages.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>P</mi><mo>(</mo><mtext>money</mtext><mo>∣</mo><mtext>spam</mtext><mo>)</mo></math>',
          text: '— likelihood of "money" in spam.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>P</mi><mo>(</mo><mtext>offer</mtext><mo>∣</mo><mtext>spam</mtext><mo>)</mo></math>',
          text: '— likelihood of "offer" in spam. Multiply all terms, then compare with the ham score.',
        },
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'How Naive Bayes learns',
      icon: 'settings',
      points: [
        'No iterative optimisation — training is **just counting**.',
        '**Prior**: count documents per class — P(spam) = spam docs / total docs.',
        '**Likelihoods**: count how often each word appears in each class — P(free | spam) = count of "free" in spam / total words in spam.',
        '**Laplace smoothing** adds 1 to every count so unseen words get a small probability instead of zero.',
        'One pass through the data — this is why NB trains in milliseconds even on large corpora.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'Naive Bayes pipeline',
      code: `from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB()),
])`,
      highlights: [
        'Fast to train — strong baseline for short text',
        'Works well with TF-IDF and count vectors',
      ],
    },
  },
  {
    type: 'title',
    content: {
      title: 'Demo - Naive Bayes on tips',
      subtitle: 'python module-03-text-classification/demo/demo.py - option 3',
      icon: 'terminal',
    },
  },

  // ---- LogisticRegression ----
  {
    type: 'title',
    content: {
      title: 'Logistic Regression',
      subtitle: 'Interpretable weights and probabilities',
      icon: 'scale',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'The sigmoid classifier',
      icon: 'trending-up',
      points: [
        'Models class probability as a **sigmoid** of a linear combination of TF-IDF features.',
        'Each feature gets a weight — **positive** pushes toward one class, **negative** pushes away.',
        'Weights are **interpretable**: inspect the top features to see which words drive predictions.',
        '**C** controls regularisation — smaller C penalises large weights, reducing overfitting on rare words.',
      ],
    },
  },
  {
    type: 'equation',
    content: {
      title: 'Logistic regression — the sigmoid',
      mathml:
        '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>P</mi><mo>(</mo><mi>y</mi><mo>=</mo><mn>1</mn><mo>∣</mo><mi mathvariant="bold">x</mi><mo>)</mo><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><mo>(</mo><mi mathvariant="bold">w</mi><mo>·</mo><mi mathvariant="bold">x</mi><mo>+</mo><mi>b</mi><mo>)</mo></mrow></msup></mrow></mfrac></math>',
      points: [
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>P</mi><mo>(</mo><mi>y</mi><mo>=</mo><mn>1</mn><mo>∣</mo><mi mathvariant="bold">x</mi><mo>)</mo></math>',
          text: '— predicted probability that input **x** belongs to class 1.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi mathvariant="bold">w</mi><mo>·</mo><mi mathvariant="bold">x</mi></math>',
          text: '— dot product of learned weights and input features (a linear score).',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>b</mi></math>',
          text: '— bias term that shifts the decision boundary.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mi>e</mi><mrow><mo>−</mo><mi>z</mi></mrow></msup></mrow></mfrac></math>',
          text: '— the sigmoid function squashes any real number into (0, 1).',
        },
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'How logistic regression learns',
      icon: 'settings',
      points: [
        'Start with random weights — predictions are terrible.',
        'Measure error with **log loss**: penalises confident wrong answers heavily.',
        'Adjust weights by **gradient descent** — nudge each weight in the direction that reduces the loss.',
        'Repeat for many iterations — `max_iter=1000` — until the weights converge.',
        'The result: a weight per word that reflects how much it pushes toward each class.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'Logistic Regression pipeline',
      code: `from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000)),
])`,
      highlights: [
        'Calibrated probability outputs via predict_proba',
        'Inspect feature weights to understand predictions',
      ],
    },
  },
  {
    type: 'title',
    content: {
      title: 'Demo - Logistic Regression on tips',
      subtitle: 'python module-03-text-classification/demo/demo.py - option 4',
      icon: 'terminal',
    },
  },

  // ---- LinearSVC ----
  {
    type: 'title',
    content: {
      title: 'Linear SVM',
      subtitle: 'Maximum-margin decision boundary',
      icon: 'shield',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Support Vector Machine',
      icon: 'maximize',
      points: [
        'Finds the **hyperplane** that **maximises the margin** between classes.',
        'On high-dimensional sparse text data, a linear separator generalises well.',
        'No native probability outputs — predicts a class label directly.',
        '**C** controls the margin-error trade-off — same regularisation idea as logistic regression.',
      ],
    },
  },
  {
    type: 'equation',
    content: {
      title: 'SVM margin',
      mathml:
        '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mtext>margin</mtext><mo>=</mo><mfrac><mn>2</mn><mrow><mo>‖</mo><mi mathvariant="bold">w</mi><mo>‖</mo></mrow></mfrac></math>',
      points: [
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mtext>margin</mtext></math>',
          text: '— the gap between the two closest points of opposite classes.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi mathvariant="bold">w</mi></math>',
          text: '— the weight vector defining the decision boundary orientation.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mo>‖</mo><mi mathvariant="bold">w</mi><mo>‖</mo></math>',
          text: '— magnitude of **w**; smaller weights → wider margin → better generalisation.',
        },
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'How SVM learns',
      icon: 'settings',
      points: [
        'Goal: find the boundary with the **widest possible margin** between classes.',
        'Formulated as an **optimisation problem** — minimise weight size while keeping all training points on the correct side.',
        '**C** relaxes the constraint: small C allows some points inside the margin (soft margin) for better generalisation.',
        'Only the **support vectors** (nearest points to the boundary) affect the solution — all other points are ignored.',
        'Much faster than trying every possible boundary — solved with efficient quadratic programming.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'Linear SVM pipeline',
      code: `from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LinearSVC(dual="auto")),
])`,
      highlights: [
        'Sharp decision boundaries on sparse text',
        'Use dual="auto" for wide TF-IDF matrices',
      ],
    },
  },
  {
    type: 'title',
    content: {
      title: 'Demo - Linear SVM on tips',
      subtitle: 'python module-03-text-classification/demo/demo.py - option 5',
      icon: 'terminal',
    },
  },

  // ---- Compare classifiers ----
  {
    type: 'title',
    content: {
      title: 'Comparing classifiers',
      subtitle: 'Which model wins on your data?',
      icon: 'bar-chart-2',
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
    type: 'title',
    content: {
      title: 'Demo - Classifier shootout',
      subtitle: 'python module-03-text-classification/demo/demo.py - option 6',
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
      mathml:
        '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mtable columnalign="right center left" rowspacing="1em"><mtr><mtd><mtext>Precision</mtext></mtd><mtd><mo>=</mo></mtd><mtd><mfrac><mtext>TP</mtext><mrow><mtext>TP</mtext><mo>+</mo><mtext>FP</mtext></mrow></mfrac></mtd></mtr><mtr><mtd><mtext>Recall</mtext></mtd><mtd><mo>=</mo></mtd><mtd><mfrac><mtext>TP</mtext><mrow><mtext>TP</mtext><mo>+</mo><mtext>FN</mtext></mrow></mfrac></mtd></mtr><mtr><mtd><msub><mtext>F</mtext><mn>1</mn></msub></mtd><mtd><mo>=</mo></mtd><mtd><mn>2</mn><mo>·</mo><mfrac><mrow><mtext>Precision</mtext><mo>·</mo><mtext>Recall</mtext></mrow><mrow><mtext>Precision</mtext><mo>+</mo><mtext>Recall</mtext></mrow></mfrac></mtd></mtr></mtable></math>',
      points: [
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mtext>TP</mtext></math>',
          text: '— true positives: correctly predicted positive instances.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mtext>FP</mtext></math>',
          text: '— false positives: negative instances incorrectly predicted as positive.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mtext>FN</mtext></math>',
          text: '— false negatives: positive instances the model missed.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><msub><mtext>F</mtext><mn>1</mn></msub></math>',
          text: '— harmonic mean of precision and recall; low if either metric is low.',
        },
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'The confusion matrix',
      code: `                 Predicted
                 credible    hoax
              ┌───────────┬──────────┐
  Actual      │           │          │
  credible    │    TN     │    FP    │
              │           │          │
              ├───────────┼──────────┤
  Actual      │           │          │
  hoax        │    FN     │    TP    │
              │           │          │
              └───────────┴──────────┘`,
      highlights: [
        '**TN** / **TP** (diagonal) = correct predictions',
        '**FP** = credible tip wrongly flagged as hoax (false alarm)',
        '**FN** = hoax that slipped through as credible (missed)',
      ],
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
      subtitle: 'python module-03-text-classification/demo/demo.py - option 7',
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
