# Exercise 02 - Tokenization

Explore **BPE subword tokenization** with tiktoken (no model download). Compare to Hugging Face **AutoTokenizer** for DistilBERT.

## Before you start

```bash
pip install -e ".[nlp,dev]"
```

tiktoken is a core dependency - Ex02 main tasks need no torch. `compare_tokenizers()` loads DistilBERT only when called.

Open `start.py`. Each `# TODO` in order.

## The data

`data/inkwell/statements.json` - 10 witness statements for token analysis.

## What you'll build

```bash
python start.py
```

## Functions

1. `load_tiktoken_encoding()` - `cl100k_base` encoding
2. `count_tokens(encoding, text)`
3. `show_subwords(encoding, text)` - BPE pieces
4. `batch_token_stats(encoding, texts)` - min/max/mean/total
5. `truncate_analysis(encoding, text, max_tokens)`
6. `compare_tokenizers(text)` - tiktoken vs AutoTokenizer side by side

## Part B - Real-world (optional)

```bash
python start.py --real-world
```

Compare mean token counts: Inkwell statements vs SMS spam sample.

## Tests

```bash
cd module-07-transformers/exercises/02-tokenization
pytest test_start.py test_extension.py -v
```
