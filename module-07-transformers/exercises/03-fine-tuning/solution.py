"""
Exercise 03 - Fine-Tuning (solution)
Requires: pip install -e ".[nlp,local-ml,dev]"
"""

import argparse
import json
from pathlib import Path

import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

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
    return json.loads(path.read_text())


def split_data(records: list[dict]) -> tuple[list[str], list[str], list[str], list[str]]:
    texts = [r["text"] for r in records]
    labels = [r["sentiment"] for r in records]
    return train_test_split(
        texts, labels, test_size=0.25, random_state=RANDOM_STATE, stratify=labels
    )


def sklearn_baseline(texts: list[str], labels: list[str]) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=RANDOM_STATE, stratify=labels
    )
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", MultinomialNB()),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 3),
        "f1_hostile": round(float(f1_score(y_test, preds, pos_label="hostile")), 3),
    }


def predict_samples(model, tokenizer, texts: list[str]) -> list[dict]:
    results = []
    model.eval()
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        pred_id = probs.argmax(dim=-1).item()
        confidence = probs[0, pred_id].item()
        results.append({
            "text_snippet": text[:60],
            "predicted_label": ID2LABEL[pred_id],
            "confidence": round(confidence, 3),
        })
    return results


def tokenize_dataset(tokenizer, texts: list[str], labels: list[str]):
    from datasets import Dataset

    label_ids = [LABEL2ID[l] for l in labels]
    ds = Dataset.from_dict({"text": texts, "label": label_ids})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    return ds.map(tokenize, batched=True)


def build_trainer(model, tokenizer, train_ds, eval_ds):
    from transformers import Trainer, TrainingArguments

    args = TrainingArguments(
        output_dir="./tmp_finetune",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=5,
        seed=RANDOM_STATE,
    )
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )


def train_and_evaluate(records: list[dict]) -> dict:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    texts = [r["text"] for r in records]
    labels = [r["sentiment"] for r in records]
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.25, random_state=RANDOM_STATE, stratify=labels
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )

    train_ds = tokenize_dataset(tokenizer, train_texts, train_labels)
    eval_ds = tokenize_dataset(tokenizer, test_texts, test_labels)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    eval_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    trainer = build_trainer(model, tokenizer, train_ds, eval_ds)
    trainer.train()
    metrics = trainer.evaluate()
    return {
        "eval_loss": round(float(metrics["eval_loss"]), 4),
        "eval_accuracy": round(float(metrics.get("eval_accuracy", 0)), 3),
        "model": model,
        "tokenizer": tokenizer,
    }


def compare_to_baseline(hf_metrics: dict, baseline_metrics: dict) -> dict:
    return {"transformer": hf_metrics, "sklearn_baseline": baseline_metrics}


def run_inkwell() -> None:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    records = load_sentiment_data(SENTIMENT_PATH)
    texts = [r["text"] for r in records]
    labels = [r["sentiment"] for r in records]

    print("Inkwell Investigations - Fine-Tuning Lab")
    print("=" * 42)

    # Before training - pre-trained model predictions
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
    )
    sample_texts = [records[0]["text"], records[1]["text"], records[3]["text"]]
    sample_actuals = [records[0]["sentiment"], records[1]["sentiment"], records[3]["sentiment"]]

    print("\n--- Before training (pre-trained DistilBERT) ---")
    before = predict_samples(model, tokenizer, sample_texts)
    for pred, actual in zip(before, sample_actuals, strict=True):
        print(f"  {pred['text_snippet'][:50]}...")
        print(f"    predicted={pred['predicted_label']} ({pred['confidence']})  actual={actual}")

    # Sklearn baseline
    baseline = sklearn_baseline(texts, labels)
    print(f"\n--- Sklearn baseline (TF-IDF + Naive Bayes) ---")
    print(f"  Accuracy: {baseline['accuracy']} | F1 (hostile): {baseline['f1_hostile']}")

    # Fine-tune
    print(f"\n--- Fine-tuning DistilBERT (3 epochs) ---")
    result = train_and_evaluate(records)

    # After training
    print(f"\n--- After training (fine-tuned) ---")
    ft_model = result.get("model", model)
    ft_tokenizer = result.get("tokenizer", tokenizer)
    after = predict_samples(ft_model, ft_tokenizer, sample_texts)
    for pred, actual in zip(after, sample_actuals, strict=True):
        match = "✓" if pred["predicted_label"] == actual else "✗"
        print(f"  {pred['text_snippet'][:50]}...")
        print(f"    predicted={pred['predicted_label']} ({pred['confidence']})  actual={actual} {match}")

    # Comparison
    hf_metrics = {"eval_loss": result["eval_loss"], "eval_accuracy": result["eval_accuracy"]}
    comparison = compare_to_baseline(hf_metrics, baseline)
    print(f"\n--- Comparison ---")
    print(f"  Sklearn baseline:  accuracy={baseline['accuracy']}  F1={baseline['f1_hostile']}")
    print(f"  Fine-tuned BERT:   accuracy={result['eval_accuracy']}  loss={result['eval_loss']}")


def run_real_world() -> None:
    reviews = json.loads(REVIEWS_PATH.read_text())
    records = [
        {"text": r["text"], "sentiment": "calm" if r["sentiment"] == "pos" else "hostile"}
        for r in reviews
    ]
    texts = [r["text"] for r in records]
    labels = [r["sentiment"] for r in records]

    print("Fine-Tuning Lab - Movie Reviews Sample")
    print("=" * 42)

    baseline = sklearn_baseline(texts, labels)
    print(f"Sklearn baseline - accuracy: {baseline['accuracy']}, F1: {baseline['f1_hostile']}")

    print("\nFine-tuning DistilBERT on movie reviews...")
    result = train_and_evaluate(records)
    print(f"DistilBERT - eval accuracy: {result['eval_accuracy']}")
    print(f"DistilBERT - eval loss: {result['eval_loss']}")


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
