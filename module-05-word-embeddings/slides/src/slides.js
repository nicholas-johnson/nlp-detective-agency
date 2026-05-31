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
      title: 'Module 5 - Word Embeddings',
      subtitle: 'A map of meaning in vector space',
      icon: 'brain',
    },
  },
  {
    type: 'welcome',
    content: {
      title: 'Aliases and code words',
      points: [
        'BoW and TF-IDF treat **synonyms as unrelated** - "accountant" vs "numbers guy".',
        'Embeddings place similar words **close together** in dense vector space.',
        'Train on case files (Word2Vec) or load pre-trained vectors (GloVe).',
        'Module 7 adds **contextual** embeddings - same word, different meaning.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Learning goals',
      icon: 'target',
      points: [
        'Understand **distributional semantics** - meaning from context.',
        'Train **Word2Vec** on Inkwell case files with gensim.',
        'Load pre-trained **GloVe** vectors for general vocabulary.',
        'Run **similarity**, **analogies**, and **vector arithmetic**.',
        'Build **document embeddings** by averaging word vectors.',
        'Optional: **OpenAI API embeddings** for semantic search.',
      ],
    },
  },

  // ---- Distributional semantics ----
  {
    type: 'title',
    content: {
      title: 'Distributional semantics',
      subtitle: 'You shall know a word by the company it keeps',
      icon: 'compass',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Sparse vs dense',
      icon: 'hash',
      points: [
        '**BoW / TF-IDF:** sparse, 10,000+ dimensions, exact word match.',
        '**Word2Vec:** dense, 50–300 dimensions, semantic similarity.',
        'Similarity = **cosine distance** between two vectors.',
        'Synonyms cluster even with **zero vocabulary overlap**.',
      ],
    },
  },

  // ---- Word2Vec ----
  {
    type: 'title',
    content: {
      title: 'Word2Vec',
      subtitle: 'Learning the map from co-occurrence',
      icon: 'brain',
    },
  },
  {
    type: 'cards',
    content: {
      title: 'CBOW vs Skip-gram',
      cards: [
        {
          title: 'CBOW',
          icon: 'layers',
          points: [
            'Predict target word from context',
            'Faster training',
            'Better on frequent words',
          ],
        },
        {
          title: 'Skip-gram',
          icon: 'zap',
          points: [
            'Predict context from target word',
            'Better on rare words',
            'Default for small corpora',
          ],
        },
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'Training with gensim',
      code: `from gensim.models import Word2Vec

sentences = [["dock", "warehouse", "reeves"], ...]
model = Word2Vec(
    sentences, vector_size=50,
    window=5, min_count=2, seed=42,
)
model.wv.most_similar("dock", topn=5)`,
      highlights: [
        'min_count drops rare noise on small corpora',
        'Train on Inkwell statements + cold cases',
      ],
    },
  },

  // ---- Pre-trained ----
  {
    type: 'title',
    content: {
      title: 'Pre-trained embeddings',
      subtitle: 'GloVe and fastText',
      icon: 'globe',
    },
  },
  {
    type: 'code',
    content: {
      title: 'Loading GloVe',
      code: `import gensim.downloader as api

model = api.load("glove-wiki-gigaword-50")
model.similarity("king", "queen")
model.most_similar(
    positive=["king", "woman"],
    negative=["man"], topn=1,
)`,
      highlights: [
        'Billions of training tokens - general vocabulary',
        'fastText handles OOV via character n-grams',
      ],
    },
  },

  // ---- Similarity ----
  {
    type: 'title',
    content: {
      title: 'Similarity and analogies',
      subtitle: 'Detective work in vector space',
      icon: 'search',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Vector arithmetic',
      icon: 'brain',
      points: [
        '**king - man + woman ≈ queen** - relational structure in dimensions.',
        '**most_similar(positive, negative)** for analogy queries.',
        '**doesnt_match([...])** finds the odd word out.',
        'Apply to alias detection: words near "accountant" in case files.',
      ],
    },
  },

  // ---- Document vectors ----
  {
    type: 'title',
    content: {
      title: 'Document embeddings',
      subtitle: 'Averaging word vectors',
      icon: 'layers',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'When averaging works',
      icon: 'scale',
      points: [
        'Average word vectors → one vector per document.',
        'Works on **short, single-topic** text (witness statements).',
        'Fails on long mixed-topic documents - opposing senses cancel out.',
        'Transformers (Module 7) solve this with contextual embeddings.',
      ],
    },
  },

  // ---- Contextual vs static ----
  {
    type: 'title',
    content: {
      title: 'Static vs contextual',
      subtitle: 'The bridge to Module 7',
      icon: 'eye',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'One vector or many?',
      icon: 'tag',
      points: [
        '**Word2Vec / GloVe:** "bank" always gets the same vector.',
        '**BERT / transformers:** "river bank" vs "bank robbery" → different vectors.',
        'Static embeddings are a strong **baseline**.',
        'Contextual embeddings are the **modern standard**.',
      ],
    },
  },

  // ---- API embeddings ----
  {
    type: 'standard',
    content: {
      title: 'API embeddings (optional)',
      icon: 'globe',
      points: [
        'OpenAI **text-embedding-3-small** - one vector per document.',
        'No OOV, captures context, 1536 dimensions.',
        'Requires API key - Exercise 03 only.',
        'Compare with TF-IDF to find semantic pairs word overlap missed.',
      ],
    },
  },

  // ---- Demo ----
  {
    type: 'title',
    content: {
      title: 'Demo - Embedding Lab',
      subtitle: 'python module-05-word-embeddings/demo/demo.py',
      icon: 'terminal',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Demo options',
      icon: 'list',
      points: [
        '**1** - Train Word2Vec, show vocab size',
        '**2** - Nearest neighbours for any word',
        '**3** - Vector arithmetic',
        '**4** - Pre-trained GloVe queries',
        '**5** - Document similarity by averaging',
        '**6** - PCA projection of top words',
      ],
    },
  },

  // ---- Wrap-up ----
  {
    type: 'title',
    content: {
      title: 'Putting it all together',
      subtitle: 'Field rules and exercises',
      icon: 'check-square',
    },
  },
  {
    type: 'rules',
    content: {
      title: 'Field rules - Module 5',
      rules: [
        {
          rule: 'Domain train vs pre-trained',
          example: 'Word2Vec on case files for jargon; GloVe for general analogies.',
          icon: 'compass',
        },
        {
          rule: 'OOV is real',
          example: 'Unseen words get no vector - fastText subwords help.',
          icon: 'alert-triangle',
        },
        {
          rule: 'Averaging is a baseline',
          example: 'Document vectors from word means work on short text only.',
          icon: 'layers',
        },
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Getting started with exercises',
      icon: 'settings',
      points: [
        'Install: **`pip install -e ".[nlp,dev]"`**.',
        'Download GloVe: **`python -c "import gensim.downloader; gensim.downloader.load(\'glove-wiki-gigaword-50\')"`**.',
        'Run tests from each exercise folder: **`pytest test_start.py -v`**.',
        'Exercises are **independent** - any order.',
      ],
    },
  },
  {
    type: 'welcome',
    content: {
      title: 'Exercises',
      points: [
        '01 - Alias Map: train Word2Vec on Inkwell, explore alias clusters',
        '02 - Embedding Compass: pre-trained GloVe similarity and analogies',
        '03 - Semantic Search: OpenAI embeddings vs TF-IDF (optional, API key)',
      ],
    },
  },
  {
    type: 'title',
    content: {
      title: 'Module 5 - Complete',
      subtitle: 'Next: Linguistic Structure - POS tagging and NER',
      icon: 'check-circle',
    },
  },
];
