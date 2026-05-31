"""
Exercise 03 - Fine-Tuning (optional)
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
    """Stratified train/test split of texts and labels."""
    # TODO
    raise NotImplementedError


def sklearn_baseline(texts: list[str], labels: list[str]) -> dict:
    """TF-IDF + Naive Bayes baseline; return accuracy and f1_hostile."""
    # TODO
    raise NotImplementedError


def tokenize_dataset(tokenizer, texts: list[str], labels: list[str]):
    """Build Hugging Face Dataset with tokenized text and label ids."""
    # TODO
    raise NotImplementedError


def build_trainer(model, tokenizer, train_ds, eval_ds):
    """Return configured transformers Trainer."""
    # TODO
    raise NotImplementedError


def train_and_evaluate(records: list[dict]) -> dict:
    """Fine-tune DistilBERT; return eval_loss and eval_accuracy."""
    # TODO
    raise NotImplementedError


def compare_to_baseline(hf_metrics: dict, baseline_metrics: dict) -> dict:
    """Return side-by-side transformer and sklearn metrics."""
    # TODO
    raise NotImplementedError


def run_inkwell() -> None:
    # TODO
    raise NotImplementedError


def run_real_world() -> None:
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
