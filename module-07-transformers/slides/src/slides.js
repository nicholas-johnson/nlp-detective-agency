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
      title: 'Module 7 - Transformers',
      subtitle: 'Contextual models and Hugging Face pipelines',
      icon: 'brain',
    },
  },
  {
    type: 'welcome',
    content: {
      title: 'The specialist arrives',
      points: [
        'Bag-of-words and static embeddings miss **context**.',
        'Transformers weigh every token against every other token.',
        '"Bank" near **river** vs **money** - different meaning.',
        'Module 7: **pipelines**, **tokenizers**, optional **fine-tuning**.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Learning goals',
      icon: 'target',
      points: [
        'Understand **attention** at a high level.',
        'Contrast **Word2Vec** (M5) with **contextual** BERT vectors.',
        'Run HF **`pipeline`** for sentiment, NER, zero-shot.',
        'Tokenize with **tiktoken** and **AutoTokenizer**.',
        'Optional: **fine-tune** DistilBERT on witness sentiment.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Transformers',
      subtitle: 'How the specialist thinks',
      icon: 'layers',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'What is a Transformer?',
      icon: 'cpu',
      points: [
        'A **neural network architecture** introduced in "Attention Is All You Need" (2017).',
        'Replaces recurrence (RNNs) with **self-attention** — processes all tokens in parallel.',
        'Learns **contextual representations** — the same word gets different vectors in different sentences.',
        'Pre-trained on massive text → **fine-tuned** on small tasks with little data.',
      ],
    },
  },
  {
    type: 'cards',
    content: {
      title: 'Examples of transformers',
      cards: [
        {
          heading: 'BERT',
          body: 'Bidirectional encoder. Great for classification, NER, QA. Reads left and right.',
        },
        {
          heading: 'GPT',
          body: 'Autoregressive decoder. Generates text left-to-right. Powers ChatGPT.',
        },
        {
          heading: 'DistilBERT',
          body: '40% smaller, 97% of BERT accuracy. We use this in exercises.',
        },
        {
          heading: 'T5 / BART',
          body: 'Encoder-decoder. Summarisation, translation, seq-to-seq tasks.',
        },
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Demo - Chatbot',
      subtitle:
        'python module-07-transformers/demo/demo.py - option 7',
      icon: 'terminal',
    },
  },

  {
    type: 'title',
    content: {
      title: 'Self Attention',
      subtitle: 'How transformers understand context',
      icon: 'zap',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Self-attention intuition',
      icon: 'zap',
      points: [
        'Each token **attends** to every other token in the context window.',
        'Stack of encoder layers builds **contextual** representations.',
        'Same word → **different vector** in different sentences.',
        'You use transformers via **pipelines** - no need to implement attention.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Queries, Keys, and Values',
      icon: 'search',
      points: [
        'Think of attention as a **search**: each token creates a **query** ("what am I looking for?").',
        'Every other token provides a **key** ("here is what I offer") and a **value** ("here is my content").',
        'The query is matched against all keys — **high match = high attention weight**.',
        'The output is a **weighted sum of values** — tokens that matter most contribute most.',
      ],
    },
  },
  {
    type: 'equation',
    content: {
      title: 'Scaled dot-product attention',
      mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mtext>Attention</mtext><mo>(</mo><mi>Q</mi><mo>,</mo><mi>K</mi><mo>,</mo><mi>V</mi><mo>)</mo><mo>=</mo><mtext>softmax</mtext><mrow><mo>(</mo><mfrac><mrow><mi>Q</mi><msup><mi>K</mi><mi>T</mi></msup></mrow><msqrt><msub><mi>d</mi><mi>k</mi></msub></msqrt></mfrac><mo>)</mo></mrow><mi>V</mi></math>',
      points: [
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>Q</mi></math>',
          text: '— queries: what each token is looking for.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>K</mi></math>',
          text: '— keys: what each token advertises about itself.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>V</mi></math>',
          text: '— values: the actual information each token contributes.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><msqrt><msub><mi>d</mi><mi>k</mi></msub></msqrt></math>',
          text: '— scaling factor: prevents dot products from growing too large in high dimensions.',
        },
        {
          mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML"><mtext>softmax</mtext></math>',
          text: '— normalises scores into attention weights that sum to 1.',
        },
      ],
      credit: 'Vaswani et al., 2017',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Static vs contextual',
      icon: 'git-compare',
      points: [
        '**Word2Vec (M5):** one vector per word, always.',
        '**Transformer:** vector depends on **surrounding context**.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Demo - Context and token IDs',
      subtitle:
        'python module-07-transformers/demo/demo.py - option 6',
      icon: 'terminal',
    },
  },

  {
    type: 'title',
    content: {
      title: 'Hugging Face pipelines',
      subtitle: 'Deploy in minutes',
      icon: 'play',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'What is Hugging Face?',
      icon: 'box',
      points: [
        'An **open-source platform** hosting 500k+ pre-trained models.',
        'The `transformers` library gives one-line access to BERT, GPT, T5, and more.',
        '**`pipeline()`** wraps tokenization, inference, and post-processing in a single call.',
        'Also hosts **datasets**, **spaces** (demos), and model cards with documentation.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Pipeline tasks',
      icon: 'list',
      points: [
        '`sentiment-analysis` - positive/negative (vs M3 classifier).',
        '`ner` - entity spans (vs M6 spaCy).',
        '`zero-shot-classification` - custom labels, **no training**.',
        '`summarization` - condense long statements.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Demo - Sentiment, NER, summarization, zero-shot',
      subtitle:
        'python module-07-transformers/demo/demo.py - options 1, 2, 3 & 4',
      icon: 'terminal',
    },
  },

  {
    type: 'title',
    content: {
      title: 'Fine-tuning (optional)',
      subtitle: 'Exercise 03',
      icon: 'settings',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'What is fine-tuning?',
      icon: 'help-circle',
      points: [
        'Start with a **pre-trained model** (DistilBERT) that already understands English.',
        'Continue training on your **small labelled dataset** (witness sentiment).',
        'The model adapts its weights to your specific task — like a specialist learning your domain.',
        'Much **less data** needed than training from scratch.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'When to fine-tune',
      icon: 'trending-up',
      points: [
        'Pipeline baseline first - know what you are improving.',
        'Fine-tune on **`witness_sentiment.json`** - calm vs hostile.',
        'Compare F1 to **TF-IDF + Naive Bayes** (Module 3).',
        'Requires `[local-ml]` - torch, datasets, accelerate.',
      ],
    },
  },

  {
    type: 'standard',
    content: {
      title: 'LoRA — lightweight fine-tuning',
      icon: 'pen-tool',
      points: [
        'Full fine-tuning updates **all** model weights — expensive and memory-heavy.',
        'LoRA freezes the original weights and injects small **trainable adapter matrices**.',
        'Typically trains **<1%** of parameters with comparable performance.',
        'Adapters are tiny files — swap tasks without duplicating the full model.',
        'Libraries: **PEFT** (HuggingFace) and **QLoRA** (quantised + LoRA).',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Demo - Fine-tuning DistilBERT',
      subtitle:
        'python module-07-transformers/demo/demo.py - option 8',
      icon: 'terminal',
    },
  },

  {
    type: 'title',
    content: {
      title: 'Evaluating language models',
      subtitle: 'How do we know it works?',
      icon: 'bar-chart',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Classification metrics',
      icon: 'target',
      points: [
        '**Accuracy** — fraction of correct predictions. Misleading on imbalanced data.',
        '**Precision** — of all positive predictions, how many were correct?',
        '**Recall** — of all actual positives, how many did we find?',
        '**F1** — harmonic mean of precision and recall. Balances both.',
        '**Macro** averages per-class scores equally; **micro** weights by class frequency.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Task-specific evaluation',
      icon: 'sliders',
      points: [
        '**NER:** span-level P/R/F1 — boundaries *and* labels must match exactly (Module 6).',
        '**Summarisation:** **ROUGE** — overlap of n-grams between generated and reference text.',
        '**Translation:** **BLEU** — precision of n-gram matches with a brevity penalty.',
        '**Language modelling:** **perplexity** — how surprised the model is by held-out text. Lower = better.',
        '**QA:** exact match (EM) and token-level F1 against gold answers.',
      ],
    },
  },
  {
    type: 'cards',
    content: {
      title: 'Standard benchmarks',
      cards: [
        {
          heading: 'GLUE / SuperGLUE',
          body: 'General Language Understanding Evaluation. Suite of 9+ tasks (sentiment, entailment, paraphrase). DistilBERT scores ~97% of BERT on GLUE.',
        },
        {
          heading: 'SQuAD',
          body: 'Stanford Question Answering Dataset. Reading comprehension QA. Model extracts answer spans from passages.',
        },
        {
          heading: 'MMLU',
          body: 'Massive Multitask Language Understanding. 57-subject multiple choice. Tests broad knowledge across domains.',
        },
        {
          heading: 'CoNLL-2003',
          body: 'Conference on Natural Language Learning. NER benchmark (Module 6). Span-level F1 is the standard metric.',
        },
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'When are benchmarks used?',
      icon: 'clock',
      points: [
        '**GLUE / SuperGLUE** — evaluating general understanding: sentiment, entailment, paraphrase.',
        '**SQuAD** — testing reading comprehension and extractive QA.',
        '**MMLU** — measuring broad knowledge across many domains (LLM evals).',
        '**CoNLL-2003** — evaluating NER: can the model find and label entity spans?',
        'Pick the benchmark that matches **your task** — or build a custom eval on your own data.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Train / validation / test splits',
      icon: 'filter',
      points: [
        'Never evaluate on training data — the model has **memorised** it.',
        '**Train** (~80%) — model learns from this.',
        '**Validation** (~10%) — tune hyperparameters, pick best checkpoint.',
        '**Test** (~10%) — final held-out evaluation. Run **once** to report results.',
        '**k-fold cross-validation** — rotate folds when data is scarce.',
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'Evaluation in practice — classification_report',
      code: `from sklearn.metrics import classification_report

y_true = ["hostile", "calm", "hostile", "calm", "calm"]
y_pred = ["hostile", "calm", "calm",    "calm", "hostile"]

print(classification_report(y_true, y_pred))

#               precision  recall  f1-score  support
#   calm           0.67     0.67     0.67        3
#   hostile        0.50     0.50     0.50        2
#   accuracy                         0.60        5
#   macro avg      0.58     0.58     0.58        5`,
      highlights: [
        'One call gives per-class **precision, recall, F1** and overall accuracy.',
        'Compare this output across models: TF-IDF baseline vs fine-tuned transformer.',
        'Exercise 01 and 03 both use this pattern to evaluate pipeline and fine-tuned results.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Beyond metrics — qualitative checks',
      icon: 'eye',
      points: [
        '**Confusion matrix** — which classes get mixed up? Spot systematic errors.',
        '**Error analysis** — read the misclassified examples. Why did the model fail?',
        '**Bias auditing** — does performance vary across demographic groups or writing styles?',
        '**Adversarial probing** — small input changes should not flip predictions.',
        'Metrics tell you **how much**; inspection tells you **why**.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Exercises',
      subtitle: 'Independent - any order',
      icon: 'book-open',
    },
  },
  {
    type: 'cards',
    content: {
      title: 'Three missions',
      cards: [
        {
          heading: '01 Inference lab',
          body: 'Sentiment, NER, zero-shot, summarisation — all local. Part B: SMS + CoNLL.',
        },
        {
          heading: '02 Text generation',
          body: 'Load distilgpt2 locally. Generate continuations, explore temperature. Part B: model comparison.',
        },
        {
          heading: '03 Fine-tuning',
          body: 'Fine-tune DistilBERT. Before/after predictions. Part B: movie reviews.',
        },
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Field rules',
      subtitle: 'Before you ship',
      icon: 'alert-triangle',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Practical guidance',
      icon: 'check-circle',
      points: [
        '**Pipeline before fine-tune** - baseline quality first.',
        '**Watch token limits** - truncate long case files (512 tokens).',
        '**Cache models** - first download is slow; then offline OK.',
        '**Ex03 (fine-tuning)** - skip if CPU/time constrained.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Bridge to Module 8',
      subtitle: 'Capstone',
      icon: 'arrow-right',
    },
  },
  {
    type: 'welcome',
    content: {
      title: 'What comes next',
      points: [
        'Module 8 chains **classical + transformer** in one pipeline.',
        'Compare TF-IDF baseline vs fine-tuned model on the big case.',
        'Persist models with **joblib** and **save_pretrained**.',
        'Expose results via a simple **FastAPI** dashboard.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Setup reminder',
      icon: 'download',
      points: [
        '`pip install -e ".[nlp,local-ml,dev]"`',
        '`python module-07-transformers/demo/demo.py`',
        '`pnpm slides:07`',
      ],
    },
  },
];
