"""
Exercise 03 - Fine-Tuning
Fine-tune DistilBERT on witness sentiment. Requires [local-ml].
"""

import argparse
import json
from pathlib import Path

SENTIMENT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "inkwell"
    / "witness_sentiment.json"
)
REVIEWS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "data"
    / "public"
    / "movie_reviews_sample.json"
)

MODEL_NAME = "distilbert-base-uncased"
RANDOM_STATE = 42
LABEL2ID = {"calm": 0, "hostile": 1}
ID2LABEL = {0: "calm", 1: "hostile"}


def load_sentiment_data(path: Path) -> list[dict]:
    """Load sentiment JSON records."""
    # TODO
    raise NotImplementedError


def split_data(records: list[dict]) -> tuple[list[str], list[str], list[str], list[str]]:
    """Stratified train/test split of texts and labels (75/25)."""
    # TODO: use sklearn train_test_split with stratify
    raise NotImplementedError


def sklearn_baseline(texts: list[str], labels: list[str]) -> dict:
    """TF-IDF + Naive Bayes baseline; return accuracy and f1_hostile."""
    # TODO: split, fit pipeline, predict, return metrics
    raise NotImplementedError


def predict_samples(model, tokenizer, texts: list[str]) -> list[dict]:
    """Run inference on sample texts using the model.

    Return list of {text_snippet, predicted_label, confidence}.
    """
    # TODO: tokenize texts, run model forward pass, apply softmax, decode labels
    raise NotImplementedError


def tokenize_dataset(tokenizer, texts: list[str], labels: list[str]):
    """Build Hugging Face Dataset with tokenized text and label ids."""
    # TODO: create Dataset, map tokenizer over it
    raise NotImplementedError


def build_trainer(model, tokenizer, train_ds, eval_ds):
    """Return configured transformers Trainer (3 epochs, batch 4, lr 5e-5)."""
    # TODO
    raise NotImplementedError


def train_and_evaluate(records: list[dict]) -> dict:
    """Fine-tune DistilBERT; return eval_loss and eval_accuracy."""
    # TODO: load model/tokenizer, split, tokenize, build trainer, train, evaluate
    raise NotImplementedError


def compare_to_baseline(hf_metrics: dict, baseline_metrics: dict) -> dict:
    """Return side-by-side transformer and sklearn metrics."""
    # TODO
    raise NotImplementedError


def run_inkwell() -> None:
    """Full pipeline: before predictions, baseline, train, after predictions, compare."""
    # TODO
    raise NotImplementedError


def run_real_world() -> None:
    """Fine-tune on movie reviews (pos=calm, neg=hostile)."""
    # TODO
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-world", action="store_true")
    args = parser.parse_args()
    if args.real_world:
        run_real_world()
    else:
        run_inkwell()


if __name__ == "__main__":
    main()
