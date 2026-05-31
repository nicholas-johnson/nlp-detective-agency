# Exercise 03 - Spam Detector

Exercise 02 compared classifiers on Inkwell tip credibility and found hoaxes that slipped through. Now run the **same shootout** on real SMS messages from the [UCI Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).

Spam that slips through as ham is the same false-negative problem as hoaxes marked credible - wasted attention on junk.

## Before you start

```bash
pip install -e ".[nlp,dev]"
```

Open `start.py`. Each function has a `# TODO` - implement them in order.

## The data

Messages live in `data/public/sms_spam_sample.json`. Each record has:

| Field   | Description                 |
| ------- | --------------------------- |
| `id`    | Message ID (e.g. `SMS-001`) |
| `label` | `ham` or `spam`             |
| `text`  | SMS message text            |

**60 messages** - 30 ham, 30 spam - sampled from the UCI corpus.

## What you'll build

```bash
python start.py
```

prints classifier comparison, confusion matrix, and slipped-through spam IDs - same format as Exercise 02.

## Functions to implement

Mirror Exercise 02 (`02-hoax-filter`) with these label changes:

| Exercise 02         | Exercise 03        |
| ------------------- | ------------------ |
| `credible` / `hoax` | `ham` / `spam`     |
| `pos_label="hoax"`  | `pos_label="spam"` |
| `load_tips`         | `load_messages`    |
| `hoax_report`       | `spam_report`      |

### Key difference for cross-validation

String labels require an explicit positive class:

```python
from sklearn.metrics import f1_score, make_scorer

spam_f1 = make_scorer(f1_score, pos_label="spam")
scores = cross_val_score(pipeline, texts, labels, cv=5, scoring=spam_f1)
```

### `false_negatives`

Return IDs where `actual == "spam"` and `predicted == "ham"`.

---

## Run it

```bash
python module-03-text-classification/exercises/03-spam-detector/start.py
```

## Run the tests

```bash
cd module-03-text-classification/exercises/03-spam-detector
pytest test_start.py -v
```

---

## Optional hints

<details>
<summary>Hint: same pipeline as hoax filter</summary>

The Inkwell tips exercise and this exercise share identical sklearn code - only the labels and dataset change. Copy the structure from `02-hoax-filter/solution.py` and adapt names.

</details>

<details>
<summary>Hint: why F1 on spam?</summary>

For a spam filter, false negatives (spam marked ham) are costly - junk reaches the user. Optimise F1 with `pos_label="spam"` to focus on catching spam.

</details>

---

## Checklist

- [ ] `load_messages` reads 60 records
- [ ] `build_pipeline` supports nb, lr, svm
- [ ] `compare_classifiers` uses `pos_label="spam"`
- [ ] `false_negatives` returns only spam misclassified as ham
- [ ] `spam_report` returns comparison, matrix, slipped IDs
- [ ] `pytest test_start.py -v` - all passed
