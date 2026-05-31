# Dataset menu - pick your case

All datasets are **real public corpora**. No synthetic or mock text. Choose one that interests you.

| Key             | Source                                                                                 | Size (approx.)  | Labels                                                        |
| --------------- | -------------------------------------------------------------------------------------- | --------------- | ------------------------------------------------------------- |
| `sms_spam`      | [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | ~5,500 messages | `ham`, `spam`                                                 |
| `newsgroups`    | sklearn `20 Newsgroups` (4 categories)                                                 | ~2,300 posts    | `sci.med`, `rec.autos`, `comp.graphics`, `talk.politics.misc` |
| `ag_news`       | [Hugging Face ag_news](https://huggingface.co/datasets/ag_news)                        | 120k train      | `World`, `Sports`, `Business`, `Sci/Tech`                     |
| `imdb`          | [Hugging Face imdb](https://huggingface.co/datasets/imdb)                              | 25k train       | `neg`, `pos`                                                  |
| `movie_reviews` | Bundled Cornell sample (`data/public/movie_reviews_sample.json`)                       | 40 reviews      | `pos`, `neg`                                                  |
| `custom`        | Your file                                                                              | varies          | your labels                                                   |

## Usage

```bash
python start.py --dataset sms_spam explore
python start.py --dataset ag_news compare --limit 5000
python start.py --dataset custom --path ~/my_data.json train
```

Use `--limit N` on large datasets (`ag_news`, `imdb`, `sms_spam`) to cap records for faster iteration on a laptop.

## Bring your own (custom)

Provide a **JSON array** or **CSV** with these columns/fields:

| Field   | Required | Description          |
| ------- | -------- | -------------------- |
| `id`    | yes      | Unique record ID     |
| `text`  | yes      | Document text        |
| `label` | yes      | Class label (string) |

**JSON example:**

```json
[
  { "id": "DOC-001", "text": "Great product, would buy again.", "label": "positive" },
  { "id": "DOC-002", "text": "Broken on arrival.", "label": "negative" }
]
```

**CSV example:**

```csv
id,text,label
DOC-001,"Great product, would buy again.",positive
DOC-002,"Broken on arrival.",negative
```

Minimum **20 records** with at least **2 classes** recommended for meaningful train/test splits.

## Tips

- **`movie_reviews`** - fastest path for development; same corpus family as Module 3/7 exercises.
- **`sms_spam`** - great for binary classification; downloads automatically on first use.
- **`newsgroups`** - multi-class, longer documents; no manual download.
- **`ag_news` / `imdb`** - larger scale; use `--limit` until your pipeline works, then remove the cap.
