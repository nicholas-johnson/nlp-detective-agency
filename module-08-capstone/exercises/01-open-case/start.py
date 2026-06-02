"""
Capstone - Open Your Case
Build an end-to-end text classifier on real data, compare models, and persist results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_PUBLIC = REPO_ROOT / "data" / "public"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

RANDOM_STATE = 42
DATASET_KEYS = ("sms_spam", "newsgroups", "ag_news", "imdb", "movie_reviews", "custom")


def list_datasets() -> list[str]:
    """Return available dataset keys."""
    # TODO
    raise NotImplementedError


def load_dataset(
    name: str,
    *,
    path: Path | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Load a dataset and normalise to {id, text, label} records."""
    # TODO
    raise NotImplementedError


def preprocess_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    # TODO
    raise NotImplementedError


def explore_dataset(records: list[dict]) -> dict:
    """Return count, class balance, length stats, and a sample record."""
    # TODO
    raise NotImplementedError


def split_records(
    records: list[dict],
    test_size: float = 0.25,
) -> tuple[list[dict], list[dict]]:
    """Stratified train/test split."""
    # TODO
    raise NotImplementedError


def build_pipeline(classifier_name: str = "lr"):
    """TF-IDF + nb/lr/svm classifier pipeline."""
    # TODO
    raise NotImplementedError


def train_baseline(train_records: list[dict], classifier_name: str = "lr"):
    """Fit baseline pipeline on training records."""
    # TODO
    raise NotImplementedError


def evaluate(model, texts: list[str], labels: list[str]) -> dict:
    """Return accuracy, f1_macro, confusion matrix, and predictions."""
    # TODO
    raise NotImplementedError


def zero_shot_predict(texts: list[str], labels: list[str]) -> list[dict]:
    """Zero-shot classify texts with HF pipeline."""
    # TODO
    raise NotImplementedError


def zero_shot_evaluate(test_records: list[dict], label_names: list[str]) -> dict:
    """Evaluate zero-shot predictions on test set."""
    # TODO
    raise NotImplementedError


def compare_models(baseline_metrics: dict, transformer_metrics: dict) -> dict:
    """Side-by-side comparison; return winner by F1 macro."""
    # TODO
    raise NotImplementedError


def error_analysis(test_records: list[dict], predictions: list[str]) -> list[dict]:
    """Return misclassified records with actual vs predicted."""
    # TODO
    raise NotImplementedError


def save_artifacts(
    pipeline,
    dataset_name: str,
    labels: list[str],
    metrics: dict,
    classifier_name: str,
    artifacts_dir: Path | None = None,
) -> Path:
    """Save joblib pipeline, config.json, and metrics.json."""
    # TODO
    raise NotImplementedError


def load_artifacts(dataset_name: str, artifacts_dir: Path | None = None):
    """Load saved pipeline and config."""
    # TODO
    raise NotImplementedError


def run_explore(dataset: str, path: Path | None, limit: int | None) -> None:
    """CLI: print dataset exploration stats."""
    # TODO
    raise NotImplementedError


def run_train(
    dataset: str,
    path: Path | None,
    limit: int | None,
    classifier: str,
) -> None:
    """CLI: train baseline and save artifacts."""
    # TODO
    raise NotImplementedError


def run_compare(
    dataset: str,
    path: Path | None,
    limit: int | None,
    classifier: str,
) -> None:
    """CLI: compare baseline vs zero-shot and print error analysis."""
    # TODO
    raise NotImplementedError


def run_serve(dataset: str, port: int) -> None:
    """CLI: start FastAPI service."""
    # TODO
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Inkwell Capstone - Open Your Case")
    parser.add_argument(
        "command",
        choices=["explore", "train", "compare", "serve", "list"],
        nargs="?",
        default="list",
    )
    parser.add_argument("--dataset", default="movie_reviews")
    parser.add_argument("--path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--classifier", choices=["nb", "lr", "svm"], default="lr")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.command == "list":
        for key in list_datasets():
            print(f"  - {key}")
        return
    if args.command == "explore":
        run_explore(args.dataset, args.path, args.limit)
    elif args.command == "train":
        run_train(args.dataset, args.path, args.limit, args.classifier)
    elif args.command == "compare":
        run_compare(args.dataset, args.path, args.limit, args.classifier)
    elif args.command == "serve":
        run_serve(args.dataset, args.port)


if __name__ == "__main__":
    main()
