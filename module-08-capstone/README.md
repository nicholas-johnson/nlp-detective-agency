# Module 8 - Applied NLP Capstone

> The big case - but this time you choose it. Every technique from the course comes together on **real data you care about**: load a public corpus or bring your own, train a classical baseline, challenge it with a transformer, understand where it breaks, persist the model, and ship a **FastAPI service** detectives (or users) can query without touching Python.

## Learning goals

- Build an **end-to-end text classification pipeline** from raw text to deployed API.
- **Choose your own dataset** from a menu of real public corpora or a custom file.
- Compare a **classical baseline** (TF-IDF + classifier) against a **transformer** (zero-shot; optional fine-tune).
- **Persist models** with joblib and run **evaluation + error analysis**.
- Expose a simple **inference API** with request logging.

## Setup

```bash
pip install -e ".[nlp,local-ml,dev]"
```

- **Baseline + API:** `[nlp]` is enough for TF-IDF training and FastAPI.
- **Zero-shot transformer:** requires PyTorch (`[local-ml]`). First run downloads models (~250 MB).
- **Optional fine-tune:** same as Module 7 Ex03 - GPU time recommended.
- **Stretch NER on errors:** `python -m spacy download en_core_web_sm`

---

## What you ship

By the end of the capstone you will have:

1. A **trained sklearn pipeline** saved to `artifacts/{dataset}/`
2. A **comparison report** - baseline vs zero-shot metrics + misclassified examples
3. A **FastAPI service** - `POST /predict` returns `{label, confidence, model}`

This is a real, runnable system - not a notebook exercise. The capstone connects every module: preprocessing (M1), TF-IDF features (M2), classifiers and evaluation (M3), zero-shot transformers (M7), and two new skills covered below: model persistence and API deployment.

---

## The capstone flow

One extended exercise walks you through eight milestones:

| Phase          | What you do                                              | Modules used  |
| -------------- | -------------------------------------------------------- | ------------- |
| 1. Pick & load | Choose a dataset; normalise to `{id, text, label}`       | -             |
| 2. Explore     | Class balance, length stats, stratified train/test split | M3            |
| 3. Preprocess  | Clean text                                               | M1            |
| 4. Baseline    | TF-IDF + nb/lr/svm; accuracy, F1, confusion matrix       | M2, M3        |
| 5. Transformer | Zero-shot with your label names; optional fine-tune      | M7            |
| 6. Compare     | Side-by-side metrics + error analysis on failures        | M3            |
| 7. Persist     | Save pipeline, config, and metrics with joblib           | (this module) |
| 8. Ship        | FastAPI `/predict` + `/health`; log every prediction     | (this module) |

See the phased checklist in [`exercises/01-open-case/README.md`](exercises/01-open-case/README.md).

Dataset options (all real data): [`exercises/01-open-case/DATASETS.md`](exercises/01-open-case/DATASETS.md).

```bash
cd module-08-capstone/exercises/01-open-case

python start.py --dataset movie_reviews explore
python start.py --dataset sms_spam train --classifier lr
python start.py --dataset newsgroups compare
python start.py --dataset sms_spam serve --port 8000
```

---

## Model persistence - filing the results

A model you cannot reload is a model you cannot deploy. After training, you need to save the pipeline so the API (or another process) can load it without retraining.

### Why joblib, not pickle?

Python's built-in `pickle` serialises any Python object, but it has problems for ML workflows:

| Issue              | pickle                                    | joblib                              |
| ------------------ | ----------------------------------------- | ----------------------------------- |
| Large numpy arrays | Slow, bloated files                       | Optimised for numpy/scipy arrays    |
| sklearn Pipeline   | Works but fragile across versions         | Recommended by scikit-learn         |
| Security           | Can execute arbitrary code on load        | Same risk - only load trusted files |
| Cross-version      | Breaks when sklearn/numpy versions change | Same - pin versions in production   |

**joblib** is scikit-learn's recommended serialisation tool. It handles the numpy arrays inside `TfidfVectorizer` and classifier objects efficiently.

### Saving artifacts

Save three things together: the model, its configuration, and evaluation metrics.

```python
import json
import joblib
from datetime import datetime, UTC
from pathlib import Path

def save_artifacts(pipeline, dataset_name, labels, metrics, classifier_name, out_dir):
    out_dir = Path(out_dir) / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # The trained model
    joblib.dump(pipeline, out_dir / "baseline.joblib")

    # Metadata - what was this model trained on?
    config = {
        "dataset": dataset_name,
        "labels": labels,
        "classifier": classifier_name,
        "saved_at": datetime.now(UTC).isoformat(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    # Evaluation results for audit
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return out_dir
```

The resulting directory structure:

```
artifacts/
  movie_reviews/
    baseline.joblib    # sklearn Pipeline (TfidfVectorizer + classifier)
    config.json        # dataset name, labels, classifier, timestamp
    metrics.json       # accuracy, F1, confusion matrix
    predictions.log    # API audit trail (created at serve time)
```

### Loading artifacts

```python
def load_artifacts(dataset_name, artifacts_dir):
    base = Path(artifacts_dir) / dataset_name
    pipeline = joblib.load(base / "baseline.joblib")
    config = json.loads((base / "config.json").read_text())
    return pipeline, config
```

Always load the config alongside the model. Six months later, you need to know which labels the model expects and which dataset it was trained on.

---

## FastAPI deployment - the detective's dashboard

Training a model in a notebook (or script) is half the job. The other half is making it **usable** - callable by other systems, other people, or a frontend - without requiring Python knowledge.

**FastAPI** is a modern Python web framework for building APIs. It validates request data with **Pydantic** models, generates automatic API documentation, and runs on **uvicorn**, a fast Python web server designed for async frameworks like FastAPI.

### Building the service

The capstone API lives in `api.py`. It loads the saved pipeline at startup and exposes two endpoints:

```python
from pathlib import Path
import json
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)

class PredictResponse(BaseModel):
    label: str
    confidence: float
    model: str

def create_app(artifacts_dir: Path) -> FastAPI:
    pipeline = joblib.load(artifacts_dir / "baseline.joblib")
    config = json.loads((artifacts_dir / "config.json").read_text())

    app = FastAPI(title="Inkwell Text Classifier")

    @app.get("/health")
    def health():
        return {"status": "ok", "dataset": config["dataset"], "labels": config["labels"]}

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest):
        label = pipeline.predict([body.text])[0]
        confidence = 1.0
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba([body.text])[0]
            classes = list(pipeline.named_steps["clf"].classes_)
            confidence = float(proba[classes.index(label)])
        return PredictResponse(label=label, confidence=round(confidence, 3),
                               model=f"tfidf-{config['classifier']}")

    return app
```

**Request flow:**

```
Client                    FastAPI                     sklearn Pipeline
  |                          |                              |
  |-- POST /predict -------->|                              |
  |   {"text": "..."}        |-- preprocess + predict ----->|
  |                          |<-- label, confidence --------|
  |<-- {"label": "pos", ...}-|                              |
```

### Running the server

```bash
cd module-08-capstone/exercises/01-open-case
python start.py train --dataset movie_reviews
python start.py serve --dataset movie_reviews --port 8000
```

Or directly with uvicorn:

```bash
uvicorn api:create_app --factory --port 8000
```

### Querying the API

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "A brilliant film with stunning performances."}'
```

Response:

```json
{ "label": "pos", "confidence": 0.87, "model": "tfidf-lr" }
```

### Testing without a running server

FastAPI provides `TestClient` for unit tests - no need to start uvicorn:

```python
from fastapi.testclient import TestClient

client = TestClient(create_app(artifacts_dir))
response = client.post("/predict", json={"text": "Terrible acting."})
assert response.status_code == 200
assert response.json()["label"] in ("pos", "neg")
```

### Audit logging

The capstone appends every prediction to `predictions.log`:

```json
{
  "timestamp": "2026-05-30T12:00:00Z",
  "text": "Terrible acting.",
  "prediction": "neg",
  "confidence": 0.91
}
```

This audit trail matters in production: when a prediction looks wrong, you can trace exactly what the model received and returned.

---

## Error analysis - where the case breaks

Aggregate metrics (accuracy, F1) tell you _how much_ the model fails. **Error analysis** tells you _why_ - and where to invest effort next.

### Step 1: Read the confusion matrix

Which classes are confused with each other? A matrix where `pos` documents are frequently classified as `neg` suggests the model struggles with subtle positive language (irony, backhanded compliments).

### Step 2: Inspect misclassified documents

Pull the actual texts the model got wrong:

```python
errors = error_analysis(test_records, predictions)
for err in errors[:10]:
    print(f"[{err['id']}] actual={err['actual']} predicted={err['predicted']}")
    print(f"  {err['text'][:200]}")
```

Look for patterns:

| Pattern           | Example                                     | Fix                                                 |
| ----------------- | ------------------------------------------- | --------------------------------------------------- |
| Class ambiguity   | Sarcastic positive review labelled negative | More labelled examples of edge cases                |
| Labelling error   | Gold label looks wrong on inspection        | Fix the dataset                                     |
| Short documents   | Three-word SMS with no signal               | Minimum length filter or different features         |
| Domain vocabulary | Jargon the model never saw                  | Domain-specific preprocessing or more training data |
| Class imbalance   | Rare class never predicted                  | Oversampling, class weights, or different metric    |

### Step 3: Decide what to try next

Error analysis closes the loop:

- **More data?** If errors are diverse, collect more labelled examples
- **Better preprocessing?** If errors contain noise (HTML, typos), improve cleaning (M1)
- **Different features?** If errors miss phrases, try bigrams (M2)
- **Different model?** If baseline fails systematically, try zero-shot or fine-tune (M7)
- **Fix labels?** If gold labels look wrong, fix the dataset before tuning the model

The capstone is not done until you can explain _where_ your system breaks and _why_.

---

## Field rules

- **Ship a baseline first.** A working TF-IDF pipeline beats a half-finished transformer.
- **Pick data you care about.** Sports news, film reviews, spam - motivation matters for error analysis.
- **Log inputs and outputs.** The API appends every prediction to `predictions.log`.
- **Error analysis is not optional.** The capstone is not done until you know where it breaks.

---

## Exercise

| Folder                                              | Mission                                              |
| --------------------------------------------------- | ---------------------------------------------------- |
| [`exercises/01-open-case`](exercises/01-open-case/) | Open your case: choose data, train, compare, deploy. |

Run tests:

```bash
pytest module-08-capstone/exercises/01-open-case/ -v
```

## Demo

Quick smoke test on bundled real data:

```bash
python module-08-capstone/demo/demo.py
```

## Slides

From repo root:

```bash
pnpm slides:capstone
```

Or: `cd module-08-capstone/slides && pnpm dev`

## Reference

- [joblib - persistence](https://joblib.readthedocs.io/en/latest/persistence.html)
- [Hugging Face - Saving and loading models](https://huggingface.co/docs/transformers/saving_models)
- [FastAPI - First Steps](https://fastapi.tiangolo.com/tutorial/)
- [FastAPI - Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- scikit-learn developers. (2024). Model persistence. _scikit-learn User Guide_.
