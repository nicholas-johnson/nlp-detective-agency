# Module 3 — Text Classification

> Tips flood in daily — some credible, some hoaxes, some planted to waste your time. Witness statements range from calm and factual to furious and unreliable. The agency needs automatic triage: models that sort leads from red herrings and flag the mood of every statement before a detective reads a word.

## Learning goals

- Split data into **train/test sets** and understand why leakage ruins evaluation.
- Build **scikit-learn pipelines** that chain vectorisation and classification.
- Train and compare classifiers: **Naive Bayes**, **Logistic Regression**, and **SVM**.
- Apply classification to **sentiment analysis** and **spam/hoax detection** on case data.
- Evaluate models with **accuracy**, **precision**, **recall**, **F1**, **confusion matrices**, and **cross-validation**.

## Setup

```bash
pip install -e ".[nlp,dev]"
```

This module uses **scikit-learn** for pipelines and classifiers. Module 2's vectorisers are composed directly into `Pipeline` objects here.

---

## Train/test splits — fair trials

A model that memorises the evidence is useless in court. Hold out a test set the model never sees during training, then evaluate on it once.

```python
from sklearn.model_selection import train_test_split

texts = [r["text"] for r in records]
labels = [r["sentiment"] for r in records]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels,
    test_size=0.25,
    random_state=42,
    stratify=labels,   # keep class proportions in both splits
)
```

**Stratify** when classes are balanced-ish — each split gets the same proportion of `calm` and `hostile` labels.

**Never** fit your vectoriser or classifier on test data. Fit on train only, evaluate on test.

---

## Pipelines — vectorise and classify in one step

Chain vectorisation and classification so they are fitted and applied together:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", MultinomialNB()),
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

The pipeline handles everything: raw text in, labels out. No separate `fit_transform` calls to forget.

---

## Classifiers for text — picking your jury

| Classifier | Strengths | Weaknesses |
| ---------- | --------- | ---------- |
| **MultinomialNB** | Fast, strong baseline for text, works well with counts/TF-IDF | Naive independence assumption |
| **LogisticRegression** | Calibrated probabilities, fast, interpretable weights | Linear decision boundary |
| **LinearSVC** | Sharp margins, strong on high-dimensional sparse text | No native probabilities |

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

Pipeline([("tfidf", TfidfVectorizer(stop_words="english")), ("clf", LogisticRegression(max_iter=1000))])
Pipeline([("tfidf", TfidfVectorizer(stop_words="english")), ("clf", LinearSVC(dual="auto"))])
```

For a first model, start with **MultinomialNB**. Compare others when you need better performance or probability estimates.

---

## Sentiment analysis — reading the room

Was the witness cooperative, hostile, or evasive? Binary sentiment classification labels the emotional tone:

- **`calm`** — factual, cooperative, neutral delivery
- **`hostile`** — defensive, angry, dismissive

```python
from sklearn.metrics import accuracy_score, f1_score

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, pos_label="hostile")
```

Check **class balance** before trusting accuracy. On a 90/10 split, a dummy classifier that always predicts the majority class scores 90% accuracy but is useless.

---

## Hoax detection — filtering the noise

Not every tip is genuine. Frame hoax detection as binary classification:

- **`credible`** — specific details, verifiable claims, named witnesses
- **`hoax`** — vague, sensational, demands for payment, conspiracy claims

For filtering tasks, **recall on hoaxes** often matters more than accuracy — missing a hoax (false negative) wastes detective hours chasing fiction.

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions, labels=["credible", "hoax"]))
```

---

## Evaluation — did the case hold up?

Accuracy alone lies. Use multiple metrics and inspect where the model fails.

```python
from sklearn.metrics import confusion_matrix, cross_val_score, f1_score, make_scorer

# Confusion matrix: rows = actual, columns = predicted
cm = confusion_matrix(y_test, predictions, labels=["credible", "hoax"])

# Cross-validation: more stable estimate on small datasets
hoax_f1 = make_scorer(f1_score, pos_label="hoax")
scores = cross_val_score(pipeline, texts, labels, cv=5, scoring=hoax_f1)
print(f"F1: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

| Metric | What it measures |
| ------ | ---------------- |
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted positives, how many are correct |
| **Recall** | Of actual positives, how many were found |
| **F1** | Harmonic mean of precision and recall |

**Inspect misclassifications.** The model's mistakes often reveal labelling issues or vocabulary gaps.

---

## Field rules

- **Optimise for the metric that matches the cost of errors.** Missing a real lead (false negative) may be worse than chasing a false one.
- **Check class balance.** Accuracy is misleading on imbalanced tip streams.
- **Inspect misclassifications.** The model's mistakes often reveal preprocessing or labelling issues.

---

## Demo

Interactive console menu — train classifiers, predict, and compare on Inkwell data:

```bash
python module-03-text-classification/demo/demo.py
```

---

## Exercises

| Folder | Mission |
| ------ | ------- |
| [`exercises/01-sentiment-triage`](exercises/01-sentiment-triage/) | Train a sentiment classifier and evaluate calm vs hostile witness statements. |
| [`exercises/02-hoax-filter`](exercises/02-hoax-filter/) | Compare classifiers on tip credibility and find hoaxes that slip through. |

Run an exercise interactively:

```bash
python module-03-text-classification/exercises/01-sentiment-triage/start.py
python module-03-text-classification/exercises/02-hoax-filter/start.py
```

Run tests (from each exercise folder):

```bash
cd module-03-text-classification/exercises/01-sentiment-triage && pytest test_start.py -v
cd module-03-text-classification/exercises/02-hoax-filter && pytest test_start.py -v
```

## Slides

From repo root: `pnpm slides:03`, or `cd module-03-text-classification/slides && pnpm dev`.

## Reference

- [scikit-learn — Text classification](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [scikit-learn — Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [scikit-learn — Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
