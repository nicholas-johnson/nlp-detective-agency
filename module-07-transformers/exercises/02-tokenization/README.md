# Exercise 02 - Text Generation

Load a small causal language model (**distilgpt2**, ~350 MB) locally and generate continuations of witness statements. Explore how temperature and sampling affect output.

## Before you start

```bash
pip install -e ".[nlp,local-ml,dev]"
```

First run downloads distilgpt2 (~350 MB, cached). Open `start.py` - each `# TODO` in order.

## The data

`data/inkwell/statements.json` - 10 witness statements to use as generation prompts.

## What you'll build

```bash
python start.py
```

```
Inkwell Investigations - Text Generation Lab
=============================================

Loading distilgpt2...

--- Continuations ---
STM-001 (Margaret Hayes):
  Prompt: "I saw HIM near the DOCKS on Tuesday!!!"
  →  "I saw HIM near the DOCKS on Tuesday!!! The man in the grey coat turned
      and looked directly at me before disappearing into the fog..."

--- Temperature variants (STM-003) ---
  temp=0.3: "Margaret Hayes is mistaken. I was at the warehouse until midnight.
             The workers confirmed my presence..."
  temp=0.7: "Margaret Hayes is mistaken. I was at the warehouse until midnight.
             But there was a sound, like someone running..."
  temp=1.2: "Margaret Hayes is mistaken. I was at the warehouse until midnight.
             Flames licked the western wall as the clock struck..."

--- Interrogation prompts ---
  Witness: Eleanor Marsh
  Context: Claims she was at the warehouse until midnight.
  Generated question: "If you were at the warehouse, can you describe what
  cargo was being loaded that evening?"
```

## Functions

1. `load_generator(model_name)` - load and cache a `text-generation` pipeline
2. `continue_statement(text, max_new_tokens)` - generate a single continuation
3. `generate_variants(text, n, temperature)` - generate `n` continuations at a given temperature
4. `interrogation_prompt(witness_name, context)` - build a prompt and generate a follow-up question
5. `batch_generate(statements, max_new_tokens)` - generate continuations for all statements

## Part B - Real-world (optional)

```bash
python start.py --real-world
```

Compare generation speed and quality between `distilgpt2` and `gpt2` (larger, ~500 MB).

## Tests

```bash
cd module-07-transformers/exercises/02-tokenization
pytest test_start.py test_extension.py -v
```

## Checklist

- [ ] `load_generator` loads distilgpt2 without errors
- [ ] `continue_statement` produces text longer than the input
- [ ] `generate_variants` produces different outputs at high temperature
- [ ] `interrogation_prompt` generates a plausible follow-up question
- [ ] All tests pass
