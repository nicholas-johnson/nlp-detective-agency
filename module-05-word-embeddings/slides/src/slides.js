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
    type: 'standard',
    content: {
      title: 'What is Word2Vec?',
      icon: 'compass',
      points: [
        'Each word becomes a **dense vector** — 50–300 numbers instead of 10,000+ sparse.',
        'Vectors are learned by training a neural network to **predict context words**.',
        'The prediction task is a means to an end — the learned weights *are* the word vectors.',
        'Words that appear in similar contexts end up with **similar vectors**.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Two ways to train Word2Vec',
      icon: 'git-branch',
      points: [
        'Word2Vec learns vectors by **predicting words from context** — but which direction?',
        'You pick **one** of two training strategies: **CBOW** or **Skip-gram**.',
        'Both use the same shallow network and backprop — only the prediction direction differs.',
        'The trained network is **disposable** — you keep the hidden layer weights as your word vectors.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'CBOW (Continuous Bag of Words)',
      icon: 'layers',
      points: [
        '"The cat **___** on the mat" — cover a word, **guess it from the neighbours**.',
        'A shallow neural network trained with **backpropagation**.',
        'The hidden layer weights *are* the embedding table — training is just a means to **learn those vectors**.',
        "Averages the context vectors — **word order doesn't matter**.",
        'Faster than Skip-gram; better on **frequent words**.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Skip-gram',
      icon: 'zap',
      points: [
        'The reverse of CBOW: start from **one word**, predict the words around it.',
        '"sat" → ("sat","the"), ("sat","cat"), ("sat","on") — pairs that **skip across** positions.',
        'Same shallow network, same backprop — but each window produces **multiple training pairs**.',
        'More work per window gives **richer signal**, especially for **rare words**.',
        'The default in most setups — Skip-gram + negative sampling (SGNS) is the standard recipe.',
      ],
    },
  },
  {
    type: 'cards',
    content: {
      title: 'CBOW vs Skip-gram',
      cards: [
        {
          heading: 'CBOW',
          points: [
            'Predict target word from context',
            'Faster training',
            'Better on frequent words',
          ],
        },
        {
          heading: 'Skip-gram',
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
    type: 'equation',
    content: {
      title: 'Skip-gram objective',
      mathml:
        '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><munder><mo>max</mo><mi>θ</mi></munder><munder><mo>∑</mo><mi>t</mi></munder><munder><mo>∑</mo><mrow><mo>−</mo><mi>c</mi><mo>≤</mo><mi>j</mi><mo>≤</mo><mi>c</mi></mrow></munder><mi>log</mi><mspace width="0.2em"/><mi>P</mi><mo>(</mo><msub><mi>w</mi><mrow><mi>t</mi><mo>+</mo><mi>j</mi></mrow></msub><mo>∣</mo><msub><mi>w</mi><mi>t</mi></msub><mo>)</mo></math>',
      points: [
        {
          mathml:
            '<math xmlns="http://www.w3.org/1998/Math/MathML"><munder><mo>max</mo><mi>θ</mi></munder></math>',
          text: '— find the network weights that maximise this expression (via backprop).',
        },
        {
          mathml:
            '<math xmlns="http://www.w3.org/1998/Math/MathML"><munder><mo>∑</mo><mi>t</mi></munder></math>',
          text: '— for every word position *t* in the corpus.',
        },
        {
          mathml:
            '<math xmlns="http://www.w3.org/1998/Math/MathML"><munder><mo>∑</mo><mrow><mo>−</mo><mi>c</mi><mo>≤</mo><mi>j</mi><mo>≤</mo><mi>c</mi></mrow></munder></math>',
          text: '— for every neighbour within window size *c* around position *t*.',
        },
        {
          mathml:
            '<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>log</mi><mspace width="0.2em"/><mi>P</mi><mo>(</mo><msub><mi>w</mi><mrow><mi>t</mi><mo>+</mo><mi>j</mi></mrow></msub><mo>∣</mo><msub><mi>w</mi><mi>t</mi></msub><mo>)</mo></math>',
          text: '— log probability of predicting context word from target word.',
        },
      ],
      credit: 'Mikolov et al., 2013',
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

  {
    type: 'title',
    content: {
      title: 'Demo - Train Word2Vec and nearest neighbours',
      subtitle: 'python module-05-word-embeddings/demo/demo.py - options 1 & 2',
      icon: 'terminal',
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
    type: 'standard',
    content: {
      title: 'What are pre-trained embeddings?',
      icon: 'download',
      points: [
        '**GloVe** (Global Vectors) — learns from the **global** co-occurrence matrix, not local windows.',
        'You get vectors for **~400k words** without training anything yourself.',
        'Use pre-trained for general vocabulary; train **Word2Vec on domain jargon** (case files).',
        '**fastText** adds character n-grams — handles unseen words by breaking them into parts.',
      ],
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

  {
    type: 'title',
    content: {
      title: 'Demo - Pre-trained GloVe queries',
      subtitle: 'python module-05-word-embeddings/demo/demo.py - option 4',
      icon: 'terminal',
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

  {
    type: 'standard',
    content: {
      title: 'Why cosine?',
      icon: 'compass',
      points: [
        'Two vectors pointing in **similar directions** are semantically similar.',
        'Cosine ignores **magnitude** (vector length) — only direction matters.',
        '**1.0** = identical direction; **0.0** = unrelated; **-1.0** = opposite.',
        'This is how `most_similar()` ranks neighbours and how we measure analogy accuracy.',
      ],
    },
  },
  {
    type: 'equation',
    content: {
      title: 'Cosine similarity',
      mathml:
        '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>cos</mi><mo>(</mo><mi>θ</mi><mo>)</mo><mo>=</mo><mfrac><mrow><mi mathvariant="bold">a</mi><mo>·</mo><mi mathvariant="bold">b</mi></mrow><mrow><mo>‖</mo><mi mathvariant="bold">a</mi><mo>‖</mo><mspace width="0.2em"/><mo>‖</mo><mi mathvariant="bold">b</mi><mo>‖</mo></mrow></mfrac></math>',
      description:
        'Two vectors pointing in similar directions have high cosine similarity. This is how most_similar() ranks neighbours and how we measure analogy accuracy.',
    },
  },
  {
    type: 'title',
    content: {
      title: 'Demo - Vector arithmetic',
      subtitle: 'python module-05-word-embeddings/demo/demo.py - option 3',
      icon: 'terminal',
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

  {
    type: 'title',
    content: {
      title: 'Demo - Document similarity by averaging',
      subtitle: 'python module-05-word-embeddings/demo/demo.py - option 5',
      icon: 'terminal',
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

  {
    type: 'standard',
    content: {
      title: 'Vector databases',
      icon: 'database',
      points: [
        '10 statements → compute cosine similarity **in memory**.',
        '10 million documents → you need a **vector database**.',
        'Store embeddings + metadata, query by **nearest-neighbour search**.',
        'Same cosine similarity underneath, with **indexing** for speed.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Vector database landscape',
      icon: 'globe',
      points: [
        '**ChromaDB** - open-source, runs locally or in Docker, good for prototyping.',
        '**Pinecone** - fully managed cloud service, scales without ops.',
        '**Weaviate** - open-source with hybrid search (vector + keyword).',
        '**Milvus / Zilliz** - open-source, built for billion-scale datasets.',
        '**Qdrant** - open-source, Rust-based, strong filtering support.',
        '**pgvector** - PostgreSQL extension — add vectors to your existing database.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Demo - PCA projection of GloVe word clusters',
      subtitle: 'python module-05-word-embeddings/demo/demo.py - option 6',
      icon: 'terminal',
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
