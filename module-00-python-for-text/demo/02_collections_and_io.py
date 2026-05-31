"""Demo 02 — Collections and I/O on Inkwell Data

Load JSON data with pathlib, explore with Counter, defaultdict, and
comprehensions. Run with:

    python module-00-python-for-text/demo/02_collections_and_io.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "inkwell"


def main() -> None:
    print("=" * 60)
    print("DEMO 2 — Collections and I/O")
    print("=" * 60)

    # --- pathlib + JSON loading ---
    path = DATA_DIR / "statements.json"
    print(f"\n--- Loading {path.name} ---")
    statements = json.loads(path.read_text())
    print(f"Loaded {len(statements)} statement records")

    # --- List comprehension: extract texts ---
    texts = [s["raw_text"] for s in statements]
    print(f"\n--- Extracted {len(texts)} raw texts ---")
    print(f"First 80 chars: {texts[0][:80]}...")

    # --- Counter: word frequencies ---
    all_words = [w.lower() for text in texts for w in text.split()]
    freq = Counter(all_words)
    print(f"\n--- Word frequencies (top 10) ---")
    print(f"Unique words: {len(freq)}")
    for word, count in freq.most_common(10):
        bar = "#" * count
        print(f"  {word:<20} {count:>3}  {bar}")

    # --- Set comprehension: unique values ---
    witnesses = {s["witness"] for s in statements}
    case_ids = {s["case_id"] for s in statements}
    print(f"\n--- Unique values ---")
    print(f"Witnesses: {sorted(witnesses)}")
    print(f"Cases:     {sorted(case_ids)}")

    # --- Dict comprehension: lookup by ID ---
    by_id = {s["id"]: s for s in statements}
    print(f"\n--- Dict lookup ---")
    first_id = statements[0]["id"]
    print(f"by_id['{first_id}']['witness'] = {by_id[first_id]['witness']}")

    # --- defaultdict: group by witness ---
    by_witness: dict[str, list[dict]] = defaultdict(list)
    for s in statements:
        by_witness[s["witness"]].append(s)

    print(f"\n--- Grouped by witness ---")
    for witness, recs in sorted(by_witness.items()):
        print(f"  {witness}: {len(recs)} statement(s)")

    # --- Load another dataset ---
    tips = json.loads((DATA_DIR / "tips.json").read_text())
    label_counts = Counter(t["label"] for t in tips)
    print(f"\n--- Tips label distribution ---")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count}")

    print("\nDone.")


if __name__ == "__main__":
    main()
