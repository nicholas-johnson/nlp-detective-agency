# Exercise 01 - Open Your Case

The capstone. Pick a **real dataset** that interests you, build a text classifier from scratch, compare classical and transformer approaches, and **persist the results**.

## Before you start

```bash
pip install -e ".[nlp,local-ml,dev]"
```

Open `start.py` - each `# TODO` maps to a milestone below. See [`DATASETS.md`](DATASETS.md) for dataset options.

## Milestones

| #   | Phase       | Functions                                            | CLI                                               |
| --- | ----------- | ---------------------------------------------------- | ------------------------------------------------- |
| 1   | Pick & load | `list_datasets()`, `load_dataset()`                  | `python start.py list`                            |
| 2   | Explore     | `explore_dataset()`, `split_records()`               | `python start.py explore --dataset movie_reviews` |
| 3   | Preprocess  | `preprocess_text()`                                  | (used by train/compare)                           |
| 4   | Baseline    | `build_pipeline()`, `train_baseline()`, `evaluate()` | `python start.py train --dataset movie_reviews`   |
| 5   | Transformer | `zero_shot_predict()`, `zero_shot_evaluate()`        | (used by compare)                                 |
| 6   | Compare     | `compare_models()`, `error_analysis()`               | `python start.py compare --dataset movie_reviews` |
| 7   | Persist     | `save_artifacts()`, `load_artifacts()`               | (called by train/compare)                         |

## Quick start (bundled real data)

Use `movie_reviews` for fast iteration - 40 real Cornell reviews already in the repo:

```bash
python start.py explore --dataset movie_reviews
python start.py train --dataset movie_reviews --classifier lr
python start.py compare --dataset movie_reviews
```

## Stretch goals

- **FastAPI service** - wrap your model in `api.py` with `POST /predict` and `GET /health`. Run with `python start.py serve --dataset movie_reviews --port 8000`.
- **Fine-tune DistilBERT** on your labels (Module 7 Ex03 pattern) - add `--fine-tune` to compare.
- **NER on errors** - run spaCy or HF NER on misclassified documents to see if entities confuse the model.
- **Remove `--limit`** on `ag_news` or `imdb` once your pipeline works on a subset.

## Tests

```bash
cd module-08-capstone/exercises/01-open-case
pytest test_start.py -v
```

Tests use real SMS messages from `data/public/sms_spam_sample.json` - not synthetic data.

## Checklist

- [ ] `list_datasets()` returns all six keys
- [ ] `load_dataset("movie_reviews")` returns normalised records
- [ ] `explore_dataset()` reports class balance and lengths
- [ ] Baseline trains and `evaluate()` returns accuracy + F1
- [ ] `compare` prints baseline vs zero-shot table
- [ ] `save_artifacts` writes `baseline.joblib`, `config.json`, `metrics.json`
- [ ] `pytest test_start.py -v` - all passed
