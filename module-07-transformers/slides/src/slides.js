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
      title: 'Attention and transformers',
      subtitle: 'How the specialist thinks',
      icon: 'layers',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Self-attention intuition',
      icon: 'zap',
      points: [
        'Each token **attends** to all others in the sentence.',
        'Stack of encoder layers builds **contextual** representations.',
        'Same word → **different vector** in different sentences.',
        'You use transformers via **pipelines** - no need to implement attention.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Static vs contextual',
      icon: 'git-compare',
      points: [
        '**Word2Vec (M5):** one vector per word, always.',
        '**BERT / DistilBERT:** vector depends on **surrounding context**.',
        '**DistilBERT:** ~40% smaller, ~97% of BERT GLUE performance.',
        'Trade-off: **accuracy vs speed vs install size**.',
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
      title: 'Tokenization',
      subtitle: 'Subwords, not whitespace',
      icon: 'hash',
    },
  },
  {
    type: 'cards',
    content: {
      title: 'Two tokenizers in this module',
      cards: [
        {
          heading: 'tiktoken',
          body: 'BPE for OpenAI models. Core dep. Fast. Exercise 02 primary.',
        },
        {
          heading: 'AutoTokenizer',
          body: 'WordPiece for DistilBERT. HF models. Comparison in Ex02.',
        },
        {
          heading: 'NLTK (M1)',
          body: 'Word-level splits. Different purpose - preprocessing.',
        },
        { heading: 'spaCy (M6)', body: 'Linguistic tokens. Rules + statistical NER.' },
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'tiktoken essentials',
      icon: 'code',
      points: [
        '`enc = tiktoken.get_encoding("cl100k_base")`',
        '`enc.encode(text)` → token IDs',
        '`enc.decode(ids)` → round-trip text',
        'Token **count** matters for API limits and truncation.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Demo - tiktoken explorer',
      subtitle:
        'python module-07-transformers/demo/demo.py - option 5',
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
    type: 'standard',
    content: {
      title: 'Zero-shot classification',
      icon: 'sparkles',
      points: [
        'Provide **candidate labels** at inference time.',
        'No `fit()` - compare to Module 3 trained models.',
        'Useful when labels are scarce or exploratory.',
        'Domain nuance may still need **fine-tuning**.',
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
          heading: '01 HF pipelines',
          body: 'Sentiment, NER, zero-shot. Part B: SMS + CoNLL.',
        },
        {
          heading: '02 Tokenization',
          body: 'tiktoken BPE + HF compare. Part B: domain stats.',
        },
        {
          heading: '03 Fine-tuning',
          body: 'DistilBERT optional. Part B: movie reviews.',
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
        '**Ex03 optional** - skip if CPU/time constrained.',
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
