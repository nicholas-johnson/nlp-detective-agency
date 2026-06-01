# Module 3 - Text Classification

> Tips flood in daily - some credible, some hoaxes, some planted to waste your time. Witness statements range from calm and factual to furious and unreliable. The agency needs automatic triage: models that sort leads from red herrings and flag the mood of every statement before a detective reads a word.

## Learning goals

- Explain what a **model** is in supervised text classification.
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

## Train/test splits - fair trials

A model that memorises the evidence is useless in court. Text classification is a **supervised learning** task. Supervised learning means learning from **labelled input-output pairs** — each training example is a text paired with a known category (e.g. `calm` / `hostile`, `credible` / `hoax`). The model uses these pairs to discover which patterns predict which label, then applies what it learned to label unseen inputs. This contrasts with **unsupervised learning** (Module 4), where there are no labels at all and the algorithm must discover structure on its own.

Because the model needs ground truth to learn what "credible" vs "hoax" looks like, you train on one subset and evaluate on held-out data the model never saw during training.

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

**Stratify** when classes are imbalanced - each split gets the same proportion of `calm` and `hostile` labels. Without stratification, a small dataset might put all hostile examples in the training set by chance.

**Never** fit your vectoriser or classifier on test data. Fit on train only, evaluate on test. Leaking test vocabulary into the vectoriser (Module 2) or tuning hyperparameters on test results produces optimistically biased metrics that fail in production.

### Overfitting

When training accuracy is high but test accuracy is low, the model has **overfit** - it memorised training quirks rather than learning general patterns. Common causes in text classification:

- Vocabulary too large relative to training set size (rare words get perfect class separation by chance)
- No regularisation on logistic regression or SVM
- Training set too small for the number of features

**Regularisation** is the primary defence against overfitting. It adds a penalty term to the model's objective function that discourages large weights. This prevents the model from relying too heavily on any single feature — for example, a rare word that perfectly separates training classes by coincidence but carries no real predictive power. The `C` parameter in logistic regression and SVM controls the trade-off: smaller `C` means stronger regularisation (more penalty on large weights, simpler model), while larger `C` allows the model to fit the training data more closely.

Other controls: reduce `max_features` in TF-IDF, collect more labelled data, or use cross-validation (below) for a more stable estimate.

---

## Pipelines - vectorise and classify in one step

Chain vectorisation and classification so they are fitted and applied together. This prevents the common bug of fitting TF-IDF on the full corpus before splitting:

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

The pipeline handles everything: raw text in, labels out. When you call `pipeline.fit(X_train, y_train)`, the vectoriser learns vocabulary from training texts only, then the classifier learns from the resulting vectors.

---

## What is a model?

A **model** in supervised text classification is a fitted pipeline that has learned which word patterns predict which labels. When you call `pipeline.fit(X_train, y_train)`, the vectoriser learns vocabulary and the classifier learns decision parameters from labelled training pairs. When you call `pipeline.predict(X_test)`, those learned parameters are applied to text the model has never seen. The Pipeline object — vectoriser plus classifier — *is* the model.

---

## Classifiers for text - picking your jury

Module 2 gave you vectors. Now you need a decision boundary - a rule that separates classes in that high-dimensional space. Three classifiers dominate text classification.

### Multinomial Naive Bayes

Naive Bayes applies **Bayes' theorem** to classify text. Before looking at any equations, the core idea is simple: start with a **prior belief** about how common each class is (e.g. most tips are credible), then **update that belief** using the evidence in the document (e.g. words like "wire $500" are strong signals for hoax). The updated belief is the **posterior** — the probability of each class after reading the text.

#### Prior, likelihood, and posterior

Bayes' theorem connects three quantities:

- **Prior** $P(c)$ — how common is class $c$ in the training data? This is the base rate (e.g. 10 credible tips out of 18 total).
- **Likelihood** $P(d \mid c)$ — how typical is document $d$ for class $c$? If hoax tips often contain words like "conspiracy" and "payment", a tip with those words has high likelihood under the hoax class.
- **Posterior** $P(c \mid d)$ — what we actually want: the probability of class $c$ **after** reading document $d$.

The denominator $P(d)$ is the same for all classes when comparing, so we often work with proportions rather than exact probabilities.

#### Bayes' theorem

$$P(c \mid d) = \frac{P(d \mid c) \, P(c)}{P(d)}$$

Read this as: **posterior = (likelihood × prior) / evidence**. The prior is the base rate; the likelihood measures how well the document fits the class; the posterior is the updated belief.

#### The naive assumption

For text, we estimate $P(w \mid c)$ from word counts in training documents of class $c$. McCallum and Nigam (1998) popularised this as **Multinomial Naive Bayes**. The **naive** assumption treats each word as independent given the class, which lets us replace the full document likelihood with a product over individual words:

$$P(d \mid c) \propto \prod_{w \in d} P(w \mid c)^{\text{count}(w, d)}$$

Despite this unrealistic independence assumption, Multinomial NB works remarkably well on text (Rennie et al., 2003). Word occurrences are sparse and partially redundant, so the independence violation hurts less than you might expect.

**Laplace smoothing** ($\alpha = 1$) prevents zero probabilities for words unseen in a class during training - without it, a single unseen word zeroes out the entire posterior.

| Strengths                     | Weaknesses                                             |
| ----------------------------- | ------------------------------------------------------ |
| Fast to train and predict     | Independence assumption is violated                    |
| Strong baseline for text      | No native feature weights to inspect                   |
| Works well with counts/TF-IDF | Struggles when word interactions matter ("not guilty") |

```python
from sklearn.naive_bayes import MultinomialNB

Pipeline([("tfidf", TfidfVectorizer(stop_words="english")), ("clf", MultinomialNB())])
```

### Logistic Regression

Where Naive Bayes multiplies word probabilities, **logistic regression** takes a different approach: every word in the vocabulary gets a learned **weight**, and the model adds up the weights of all words present in a document to produce a single score. A high positive score means the document looks like one class; a high negative score means it looks like the other. The **sigmoid** function then converts that raw score into a probability between 0 and 1.

#### How weights combine

Imagine two TF-IDF features with learned weights: "payment" has weight +2.1 (pushes toward hoax) and "witness" has weight −1.8 (pushes toward credible). A tip containing both words gets a combined score of roughly $2.1 + (-1.8) = +0.3$ (simplified — in practice every word in the document contributes). The sigmoid converts +0.3 into a probability just above 0.5 — a marginal hoax prediction. A tip with "payment" but not "witness" scores higher and gets a more confident hoax label.

This is the linear combination $\mathbf{w} \cdot \mathbf{x} + b$, where $\mathbf{x}$ is the TF-IDF vector and $\mathbf{w}$ is the weight vector learned during training.

#### The sigmoid

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w} \cdot \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w} \cdot \mathbf{x} + b)}}$$

The **sigmoid** ($\sigma$) squashes any real number into the range (0, 1). A large positive input maps to ≈1 (confident positive prediction), a large negative input maps to ≈0 (confident negative), and zero maps to exactly 0.5 (maximum uncertainty). This is why "logistic regression" is used for classification despite the name "regression" — the sigmoid turns a continuous score into a class probability.

#### Why it is interpretable

Each TF-IDF feature gets a weight $w_i$. A **positive weight** means that word pushes toward the positive class; a **negative weight** pushes away. You can rank features by weight to see exactly which words drive a prediction — a key advantage over Naive Bayes, which has no single set of comparable feature weights. In the demo (option 4), you can inspect the top features for any tip and see *why* the model classified it as credible or hoax.

#### Regularisation and C

The **C** parameter controls regularisation (see overfitting section above): smaller C means stronger penalty on large weights, reducing overfitting on rare words. **`max_iter`** sets the optimisation iteration limit — increase it if the solver warns about non-convergence.

| Strengths                      | Weaknesses                                         |
| ------------------------------ | -------------------------------------------------- |
| Calibrated probability outputs | Linear decision boundary                           |
| Interpretable feature weights  | Needs more data than NB                            |
| Fast on sparse text features   | Sensitive to feature scaling (TF-IDF handles this) |

```python
from sklearn.linear_model import LogisticRegression

Pipeline([("tfidf", TfidfVectorizer(stop_words="english")),
          ("clf", LogisticRegression(max_iter=1000))])
```

### Linear Support Vector Machine

Logistic regression computes a probability; a **Support Vector Machine** takes a different approach. Instead of asking "how probable is each class?", SVM asks "where should I draw a dividing line between classes, and how can I make that line as robust as possible?"

#### The decision boundary

SVM draws a **decision boundary** between classes and positions it to be as far as possible from the nearest training examples on both sides. The gap between the boundary and those closest points is the **margin**.

In two dimensions (two features), the boundary is a line. In three dimensions, it is a flat plane. With thousands of TF-IDF features, it is a **hyperplane** — the same concept extended to high-dimensional space. You cannot visualise a hyperplane in thousands of dimensions, but the mathematics works identically to the 2D case.

#### Why margin matters

A boundary that barely squeezes between classes is fragile — a slightly different test document could easily land on the wrong side. A **wide margin** means the model is more confident about the gap between classes and generalises better to unseen text.

$$\text{margin} = \frac{2}{\|\mathbf{w}\|}$$

The formula shows that **smaller weights produce a wider margin** — the same regularisation intuition as logistic regression. Keeping weights small forces the model to find a robust separator rather than memorising quirks of the training data.

#### Support vectors

The training examples that sit exactly on the edge of the margin are called **support vectors**. Only these closest points determine where the boundary goes — all other training examples could be moved or removed without changing the decision boundary at all. This is why SVM works well on sparse text data: most documents are far from the boundary and do not influence it. Joachims (1998) demonstrated this effectiveness on text classification tasks.

#### No native probabilities

SVM predicts a class label, not a probability. If you need calibrated probabilities, use logistic regression instead or set `probability=True` on `SVC` (which adds a separate calibration step and is slower). For filtering tasks where you only need a binary decision (credible or hoax), the lack of probabilities is rarely a problem.

#### The C parameter

**C** controls the trade-off between margin width and training accuracy. Small C allows some misclassifications in exchange for a wider margin — more generalisation, less overfitting. Large C forces the boundary to classify all training points correctly even if the margin shrinks. On high-dimensional text data, the default or a moderately small C usually works well.

| Strengths                              | Weaknesses                                  |
| -------------------------------------- | ------------------------------------------- |
| Sharp decision boundaries              | No native probability outputs               |
| Strong on high-dimensional sparse text | Less interpretable than logistic regression |
| Fast with `LinearSVC(dual="auto")`     | Needs careful regularisation (C parameter)  |

```python
from sklearn.svm import LinearSVC

Pipeline([("tfidf", TfidfVectorizer(stop_words="english")),
          ("clf", LinearSVC(dual="auto"))])
```

For a first model, start with **MultinomialNB**. Compare logistic regression and SVM when you need better performance or probability estimates.

---

## Sentiment analysis - reading the room

Was the witness cooperative, hostile, or evasive? Binary sentiment classification labels the emotional tone:

- **`calm`** - factual, cooperative, neutral delivery
- **`hostile`** - defensive, angry, dismissive

```python
from sklearn.metrics import accuracy_score, f1_score

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, pos_label="hostile")
```

Check **class balance** before trusting accuracy. On a 90/10 split, a dummy classifier that always predicts the majority class scores 90% accuracy but is useless.

---

## Hoax detection - filtering the noise

Not every tip is genuine. Frame hoax detection as binary classification:

- **`credible`** - specific details, verifiable claims, named witnesses
- **`hoax`** - vague, sensational, demands for payment, conspiracy claims

For filtering tasks, **recall on hoaxes** often matters more than accuracy - missing a hoax (false negative) wastes detective hours chasing fiction.

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions, labels=["credible", "hoax"]))
```

---

## Evaluation - did the case hold up?

Accuracy alone lies. A confusion matrix reveals where the model succeeds and fails:

|                    | Predicted **credible** | Predicted **hoax** |
| ------------------ | ---------------------- | ------------------ |
| Actual **credible** | TN                     | FP                 |
| Actual **hoax**     | FN                     | TP                 |

From this table, derive the metrics that matter for your task:

$$\text{Precision} = \frac{TP}{TP + FP} \qquad \text{Recall} = \frac{TP}{TP + FN}$$

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **Precision** - of everything the model flagged as hoax, how many actually were? High precision means few false alarms.
- **Recall** - of all actual hoaxes, how many did the model catch? High recall means few slip through.
- **F1** - harmonic mean of precision and recall; balances both when you care equally about false positives and false negatives.

**Worked example:** 20 tips, 5 hoaxes. Model predicts 4 hoaxes correctly (TP=4), misses 1 (FN=1), and falsely flags 2 credible tips (FP=2).

- Precision = 4 / (4 + 2) = 0.667
- Recall = 4 / (4 + 1) = 0.800
- F1 = 2 × (0.667 × 0.800) / (0.667 + 0.800) = 0.727

```python
from sklearn.metrics import confusion_matrix, cross_val_score, f1_score, make_scorer

cm = confusion_matrix(y_test, predictions, labels=["credible", "hoax"])

hoax_f1 = make_scorer(f1_score, pos_label="hoax")
scores = cross_val_score(pipeline, texts, labels, cv=5, scoring=hoax_f1)
print(f"F1: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### Cross-validation

A single train/test split can be lucky or unlucky - especially on small datasets. **K-fold cross-validation** gives a more stable estimate:

1. Split data into $k$ equal folds (typically $k=5$)
2. For each fold: train on the other $k-1$ folds, test on the held-out fold
3. Average the $k$ scores

**Stratified** k-fold preserves class proportions in each fold - essential for imbalanced tip streams. Cross-validation uses all data for both training and testing (just not simultaneously), making better use of small labelled sets.

| Metric        | What it measures                             |
| ------------- | -------------------------------------------- |
| **Accuracy**  | Overall correct predictions                  |
| **Precision** | Of predicted positives, how many are correct |
| **Recall**    | Of actual positives, how many were found     |
| **F1**        | Harmonic mean of precision and recall        |

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

| Option | What it does |
| ------ | ------------ |
| 1 | List datasets (sentiment + tips class counts) |
| 2 | Sentiment pipeline — predict one witness statement |
| 3 | Naive Bayes — predict one tip + posterior probabilities |
| 4 | Logistic Regression — predict one tip + top weighted features |
| 5 | Linear SVM — predict one tip + hold-out F1 |
| 6 | Classifier shootout (NB / LR / SVM via cross-validation) |
| 7 | Classification report + confusion matrix |

Options 3–5 introduce each classifier individually before option 6 compares them.

---

## Exercises

| Folder                                                            | Mission                                                                           |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [`exercises/01-sentiment-triage`](exercises/01-sentiment-triage/) | Train a sentiment classifier and evaluate calm vs hostile witness statements.     |
| [`exercises/02-hoax-filter`](exercises/02-hoax-filter/)           | Compare classifiers on tip credibility and find hoaxes that slip through.         |
| [`exercises/03-spam-detector`](exercises/03-spam-detector/)       | Run the classifier shootout on real SMS spam and find messages that slip through. |

Run an exercise interactively:

```bash
python module-03-text-classification/exercises/01-sentiment-triage/start.py
python module-03-text-classification/exercises/02-hoax-filter/start.py
python module-03-text-classification/exercises/03-spam-detector/start.py
```

Run tests (from each exercise folder):

```bash
cd module-03-text-classification/exercises/01-sentiment-triage && pytest test_start.py -v
cd module-03-text-classification/exercises/02-hoax-filter && pytest test_start.py -v
cd module-03-text-classification/exercises/03-spam-detector && pytest test_start.py -v
```

## Slides

From repo root: `pnpm slides:03`, or `cd module-03-text-classification/slides && pnpm dev`.

## Reference

- [scikit-learn - Text classification](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [scikit-learn - Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [scikit-learn - Model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- McCallum, A., & Nigam, K. (1998). A comparison of event models for naive Bayes text classification. _AAAI-98 Workshop on Learning for Text Categorization_.
- Joachims, T. (1998). Text categorization with support vector machines. _ECML_, 137–142.
- Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003). Tackling the poor assumptions of naive Bayes text classifiers. _ICML_, 616–623.
