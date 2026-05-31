export const slides = [
  {
    type: 'title',
    content: {
      title: 'Python NLP',
      subtitle: 'Inkwell Investigations - Nicholas Johnson',
      icon: 'search',
    },
  },
  {
    type: 'title',
    content: {
      title: 'Module 8 - Applied NLP Capstone',
      subtitle: 'Open your case. Ship something real.',
      icon: 'rocket',
    },
  },
  {
    type: 'welcome',
    content: {
      title: 'The big case - your choice',
      points: [
        'Seven modules of tools. One **capstone** to tie them together.',
        'Pick a **real dataset** you care about - spam, news, reviews.',
        'Train a **baseline**, challenge it with a **transformer**.',
        'Persist the model and **deploy FastAPI**.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Learning goals',
      icon: 'target',
      points: [
        'End-to-end **text classification** pipeline.',
        'Compare **TF-IDF + sklearn** vs **zero-shot HF**.',
        '**Evaluation + error analysis** on real failures.',
        '**joblib** persistence and **FastAPI** inference.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Course arc recap',
      subtitle: 'Everything leads here',
      icon: 'layers',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'From tokens to deployment',
      icon: 'git-branch',
      points: [
        '**M1–2:** preprocess, TF-IDF features.',
        '**M3:** train classifiers, evaluate metrics.',
        '**M4–5:** topics, embeddings - context for trade-offs.',
        '**M6:** spaCy structure; **M7:** transformers.',
        '**M8:** you assemble and **ship**.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Eight milestones',
      subtitle: 'One extended exercise',
      icon: 'list-checks',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'The capstone flow',
      icon: 'arrow-right',
      points: [
        '1. **Pick & load** - choose dataset, normalise schema.',
        '2. **Explore** - balance, lengths, train/test split.',
        '3. **Preprocess** - clean text (Module 1).',
        '4. **Baseline** - TF-IDF + nb/lr/svm.',
        '5. **Transformer** - zero-shot with your label names.',
        '6. **Compare** - metrics + misclassified examples.',
        '7. **Persist** - joblib + config + metrics JSON.',
        '8. **Ship** - FastAPI `/predict` + audit log.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Choose your dataset',
      subtitle: 'All real public corpora',
      icon: 'database',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Dataset menu',
      icon: 'folder-open',
      points: [
        '**sms_spam** - UCI SMS (~5.5k messages).',
        '**newsgroups** - sklearn 4-category subset.',
        '**ag_news / imdb** - Hugging Face (use `--limit`).',
        '**movie_reviews** - bundled Cornell sample for dev.',
        '**custom** - your JSON/CSV with `{id, text, label}`.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Baseline first',
      subtitle: 'Ship a working pipeline',
      icon: 'shield',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Classical stack',
      icon: 'cpu',
      points: [
        '**TfidfVectorizer** + **LogisticRegression** (default).',
        'Also **Naive Bayes** and **Linear SVM**.',
        'Metrics: accuracy, **F1 macro**, confusion matrix.',
        'A trained baseline beats a half-finished BERT.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Zero-shot transformer',
      icon: 'brain',
      points: [
        'HF **`zero-shot-classification`** with your label names.',
        'No training required - good for quick comparison.',
        'Optional **fine-tune** DistilBERT (Module 7 stretch).',
        'Compare **F1 macro** - know when each approach wins.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Error analysis',
      subtitle: 'Where the case breaks',
      icon: 'alert-triangle',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Inspect failures',
      icon: 'search',
      points: [
        'List **misclassified** documents with actual vs predicted.',
        'Look for patterns: length, slang, class imbalance.',
        'Stretch: run **NER** on errors - do entities confuse the model?',
        'Error analysis is **required** - not optional polish.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Persist and deploy',
      subtitle: 'Make it reusable',
      icon: 'save',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Artifacts',
      icon: 'package',
      points: [
        '`baseline.joblib` - sklearn pipeline.',
        '`config.json` - dataset, labels, classifier.',
        '`metrics.json` - evaluation results.',
        '`predictions.log` - API audit trail.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'FastAPI service',
      icon: 'globe',
      points: [
        '`GET /health` - status + label set.',
        '`POST /predict` - `{"text": "..."}` → label + confidence.',
        'Load model **at startup** from artifacts dir.',
        'Log every prediction for traceability.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Field rules',
      subtitle: 'Capstone discipline',
      icon: 'book-open',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Before you call it done',
      icon: 'check-circle',
      points: [
        '**Baseline first** - working TF-IDF pipeline deployed.',
        '**Pick data you care about** - motivation drives analysis.',
        '**Log I/O** - predictions.log on every API call.',
        '**Know your failure modes** - error analysis complete.',
      ],
    },
  },
  {
    type: 'title',
    content: {
      title: 'You built something real',
      subtitle: 'Open case → closed deployment',
      icon: 'award',
    },
  },
];
