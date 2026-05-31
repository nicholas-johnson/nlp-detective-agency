"""
Inkwell Investigations - Capstone Demo
Run:  python module-08-capstone/demo/demo.py
"""

import sys
from pathlib import Path

EXERCISE_DIR = Path(__file__).resolve().parent.parent / "exercises" / "01-open-case"
sys.path.insert(0, str(EXERCISE_DIR))

import solution as capstone  # noqa: E402


def menu_list() -> None:
    print("\n--- Available datasets ---")
    for key in capstone.list_datasets():
        print(f"  {key}")


def menu_explore() -> None:
    records = capstone.load_dataset("movie_reviews")
    stats = capstone.explore_dataset(records)
    print("\n--- Explore movie_reviews (bundled real data) ---")
    print(f"Records: {stats['count']}")
    print(f"Classes: {stats['class_balance']}")
    print(f"Avg length: {stats['avg_length']} chars")


def menu_train_compare() -> None:
    records = capstone.load_dataset("movie_reviews")
    train, test = capstone.split_records(records)
    pipeline = capstone.train_baseline(train, "lr")
    test_texts = [capstone.preprocess_text(r["text"]) for r in test]
    test_labels = [r["label"] for r in test]
    metrics = capstone.evaluate(pipeline, test_texts, test_labels)
    print("\n--- Train baseline on movie_reviews ---")
    print(f"Test accuracy: {metrics['accuracy']}")
    print(f"F1 macro: {metrics['f1_macro']}")


def menu_serve_hint() -> None:
    print("\n--- Ship it ---")
    print("Train and save artifacts, then serve:")
    print("  cd module-08-capstone/exercises/01-open-case")
    print("  python solution.py train --dataset movie_reviews")
    print("  python solution.py serve --dataset movie_reviews --port 8000")
    print("\nQuery:")
    print('  curl -X POST http://127.0.0.1:8000/predict \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"text": "A brilliant film."}\'')


def main() -> None:
    options = {
        "1": ("List datasets", menu_list),
        "2": ("Explore movie_reviews", menu_explore),
        "3": ("Train baseline (movie_reviews)", menu_train_compare),
        "4": ("Serve instructions", menu_serve_hint),
    }
    print("Inkwell Capstone Demo")
    print("=" * 40)
    for key, (label, _) in options.items():
        print(f"  {key}. {label}")
    print("  q. Quit")

    choice = input("\nChoice: ").strip().lower()
    if choice == "q":
        return
    if choice in options:
        options[choice][1]()
    else:
        print("Unknown choice.")


if __name__ == "__main__":
    main()
