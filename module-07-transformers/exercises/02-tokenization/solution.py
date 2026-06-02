"""
Exercise 02 - Text Generation (solution)
"""

import argparse
import json
import time
from pathlib import Path

from transformers import pipeline, set_seed

STATEMENTS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "inkwell" / "statements.json"
)

DEFAULT_MODEL = "distilgpt2"

_generators: dict = {}


def load_generator(model_name: str = DEFAULT_MODEL):
    if model_name not in _generators:
        print(f"  Loading {model_name}...")
        _generators[model_name] = pipeline(
            "text-generation", model=model_name, pad_token_id=50256
        )
    return _generators[model_name]


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def continue_statement(text: str, max_new_tokens: int = 50) -> str:
    gen = load_generator()
    result = gen(
        text,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        num_return_sequences=1,
    )
    full = result[0]["generated_text"]
    return full[len(text):].strip()


def generate_variants(text: str, n: int = 3, temperature: float = 0.7) -> list[str]:
    gen = load_generator()
    results = gen(
        text,
        max_new_tokens=50,
        do_sample=True,
        temperature=temperature,
        num_return_sequences=n,
    )
    return [r["generated_text"][len(text):].strip() for r in results]


def interrogation_prompt(witness_name: str, context: str, max_new_tokens: int = 40) -> str:
    prompt = (
        f"Detective's follow-up question for {witness_name}, "
        f"who said: '{context}'\nQuestion:"
    )
    gen = load_generator()
    result = gen(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.8,
        num_return_sequences=1,
    )
    full = result[0]["generated_text"]
    return full[len(prompt):].strip()


def batch_generate(statements: list[dict], max_new_tokens: int = 40) -> list[dict]:
    gen = load_generator()
    results = []
    for stmt in statements:
        prompt = stmt["raw_text"][:80]
        out = gen(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            num_return_sequences=1,
        )
        continuation = out[0]["generated_text"][len(prompt):].strip()
        results.append({
            "id": stmt["id"],
            "witness": stmt["witness"],
            "prompt_snippet": prompt,
            "continuation": continuation,
        })
    return results


def run_inkwell() -> None:
    statements = load_json(STATEMENTS_PATH)

    print("Inkwell Investigations - Text Generation Lab")
    print("=" * 46)

    # Single continuation
    t0 = time.time()
    stmt = statements[0]
    prompt = stmt["raw_text"][:80]
    continuation = continue_statement(prompt)
    print(f"\n--- Continuation ({stmt['id']} {stmt['witness']}) ---")
    print(f"  Prompt: \"{prompt}...\"")
    print(f"  → \"{continuation[:120]}...\"")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Temperature variants
    t0 = time.time()
    stmt = statements[2]
    prompt = stmt["raw_text"][:60]
    print(f"\n--- Temperature variants ({stmt['id']} {stmt['witness']}) ---")
    for temp in [0.3, 0.7, 1.2]:
        variants = generate_variants(prompt, n=1, temperature=temp)
        print(f"  temp={temp}: \"{variants[0][:100]}...\"")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Interrogation
    t0 = time.time()
    stmt = statements[2]
    question = interrogation_prompt(stmt["witness"], stmt["raw_text"][:100])
    print(f"\n--- Interrogation prompt ---")
    print(f"  Witness: {stmt['witness']}")
    print(f"  Generated question: \"{question[:120]}\"")
    print(f"  Time: {time.time() - t0:.1f}s")

    # Batch
    t0 = time.time()
    batch = batch_generate(statements[:5])
    print(f"\n--- Batch generation (first 5 statements) ---")
    for r in batch:
        print(f"  {r['id']} {r['witness']}: \"{r['continuation'][:80]}...\"")
    print(f"  Time: {time.time() - t0:.1f}s")


def run_real_world() -> None:
    statements = load_json(STATEMENTS_PATH)
    prompt = statements[0]["raw_text"][:80]

    print("Text Generation - Model Comparison")
    print("=" * 36)

    for model_name in ["distilgpt2", "gpt2"]:
        t0 = time.time()
        gen = load_generator(model_name)
        load_time = time.time() - t0

        t0 = time.time()
        result = gen(prompt, max_new_tokens=50, do_sample=True, temperature=0.7)
        gen_time = time.time() - t0

        continuation = result[0]["generated_text"][len(prompt):]
        print(f"\n--- {model_name} ---")
        print(f"  Load: {load_time:.1f}s | Generate: {gen_time:.2f}s")
        print(f"  Output: \"{continuation.strip()[:120]}...\"")


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
