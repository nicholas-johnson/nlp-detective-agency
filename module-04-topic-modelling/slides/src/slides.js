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
      title: 'Module 4 - Topic Modelling',
      subtitle: 'Patterns in the cold-case archive',
      icon: 'layers',
    },
  },
  {
    type: 'welcome',
    content: {
      title: 'Let the archive speak',
      points: [
        'Module 3 **labelled** text with known categories.',
        'Now we discover themes **without labels** - unsupervised structure.',
        'Cases that look unrelated may share vocabulary: docks, ledgers, motorcars.',
        'Topic models pin recurring threads to the **evidence board**.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Learning goals',
      icon: 'target',
      points: [
        'Understand **unsupervised theme discovery** vs manual categorisation.',
        'Apply **LDA** on count features to find latent topics.',
        'Apply **NMF** on TF-IDF as an alternative lens.',
        'Choose **k topics** with perplexity and top-word inspection.',
        '**Interpret, label, and visualise** topics for briefings.',
      ],
    },
  },

  // ---- Unsupervised ----
  {
    type: 'title',
    content: {
      title: 'Unsupervised discovery',
      subtitle: 'Soft mixtures, not hard categories',
      icon: 'brain',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Topics are not labels',
      icon: 'tag',
      points: [
        'A document can belong to **several topics at once**.',
        'Weights sum to ~1 per document - a **mixture**, not a single bucket.',
        'No ground truth - topics are **hypotheses** for detectives to validate.',
        'Preprocessing (stopwords, min_df) shapes what themes emerge.',
      ],
    },
  },

  // ---- DTM ----
  {
    type: 'title',
    content: {
      title: 'Document-term matrices',
      subtitle: 'Two vectorisers, two models',
      icon: 'hash',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Count vs TF-IDF input',
      icon: 'hash',
      points: [
        '**LDA** → `CountVectorizer` - models word **counts**.',
        '**NMF** → `TfidfVectorizer` - non-negative **weighted** features.',
        '`min_df=2` drops rare one-off terms that add noise.',
        '`stop_words="english"` removes the, was, and, ...',
      ],
    },
  },

  // ---- LDA ----
  {
    type: 'title',
    content: {
      title: 'Latent Dirichlet Allocation',
      subtitle: 'Documents as mixtures of topics',
      icon: 'layers',
    },
  },
  {
    type: 'code',
    content: {
      title: 'LDA in scikit-learn',
      code: `from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer(stop_words="english", min_df=2)
dtm = vec.fit_transform(texts)

lda = LatentDirichletAllocation(
    n_components=4, random_state=42, max_iter=20,
)
doc_topics = lda.fit_transform(dtm)  # (n_docs, n_topics)`,
      highlights: [
        'components_ - topic-word distributions',
        'fit_transform - document-topic weights per file',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Demo - Archive stats and LDA top words',
      subtitle:
        'python module-04-topic-modelling/demo/demo.py - options 1 & 2',
      icon: 'terminal',
    },
  },

  // ---- NMF ----
  {
    type: 'title',
    content: {
      title: 'Non-negative Matrix Factorization',
      subtitle: 'Sharper topics on TF-IDF',
      icon: 'filter',
    },
  },
  {
    type: 'cards',
    content: {
      title: 'LDA vs NMF',
      cards: [
        {
          title: 'LDA',
          icon: 'layers',
          points: [
            'Probabilistic generative model',
            'Input: word counts',
            'Soft mixtures per document',
          ],
        },
        {
          title: 'NMF',
          icon: 'hash',
          points: [
            'Matrix factorisation',
            'Input: TF-IDF (non-negative)',
            'Often sharper top words',
          ],
        },
      ],
    },
  },
  {
    type: 'code',
    content: {
      title: 'NMF in scikit-learn',
      code: `from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer(stop_words="english", min_df=2)
tfidf = vec.fit_transform(texts)

nmf = NMF(n_components=4, random_state=42, max_iter=200)
doc_topics = nmf.fit_transform(tfidf)`,
      highlights: [
        'Compare top words with LDA on the same corpus',
        'Neither model wins every time - inspect both',
      ],
    },
  },

  {
    type: 'equation',
    content: {
      title: 'NMF factorisation',
      mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mi>V</mi><mo>≈</mo><mi>W</mi><mo>×</mo><mi>H</mi></math>',
      description: "Decompose the document-term matrix V into W (document-topic weights) and H (topic-word distributions). Both are non-negative, so weights are interpretable as 'amounts'.",
    },
  },
  {
    type: 'title',
    content: {
      title: 'Demo - Document-topic profile and NMF top words',
      subtitle:
        'python module-04-topic-modelling/demo/demo.py - options 3 & 4',
      icon: 'terminal',
    },
  },

  // ---- Choosing k ----
  {
    type: 'title',
    content: {
      title: 'Choosing the number of topics',
      subtitle: 'Perplexity plus human judgement',
      icon: 'sliders',
    },
  },
  {
    type: 'code',
    content: {
      title: 'Perplexity sweep',
      code: `for k in range(3, 7):
    lda = LatentDirichletAllocation(
        n_components=k, random_state=42,
    )
    lda.fit(dtm)
    print(k, lda.perplexity(dtm))  # lower is better`,
      highlights: [
        'Perplexity guides k - top words confirm it',
        'Too many topics → noise; too few → merged themes',
      ],
    },
  },

  {
    type: 'equation',
    content: {
      title: 'Perplexity',
      mathml: '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block"><mtext>perplexity</mtext><mo>=</mo><mi>exp</mi><mrow><mo>(</mo><mo>−</mo><mfrac><mrow><munder><mo>∑</mo><mi>d</mi></munder><mi>log</mi><mspace width="0.2em"/><mi>p</mi><mo>(</mo><msub><mi mathvariant="bold">w</mi><mi>d</mi></msub><mo>)</mo></mrow><mrow><munder><mo>∑</mo><mi>d</mi></munder><msub><mi>N</mi><mi>d</mi></msub></mrow></mfrac><mo>)</mo></mrow></math>',
      description: "Lower perplexity = the model is less 'surprised' by the data. Use as a guide for choosing k, but always check top words too — perplexity alone can't tell you if topics make sense.",
    },
  },
  {
    type: 'title',
    content: {
      title: 'Demo - Perplexity sweep',
      subtitle:
        'python module-04-topic-modelling/demo/demo.py - option 5',
      icon: 'terminal',
    },
  },

  // ---- Interpretation ----
  {
    type: 'title',
    content: {
      title: 'Interpreting topics',
      subtitle: 'From word lists to evidence board',
      icon: 'eye',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Three-step briefing',
      icon: 'clipboard-list',
      points: [
        '1. Print **top 8 words** per topic from `components_`.',
        '2. Find each case\'s **dominant topic** - argmax of document weights.',
        '3. **Label topics** manually: Waterfront, Financial, Surveillance, ...',
        'Group case IDs under each label for the evidence board.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'pyLDAvis - optional visualisation',
      icon: 'eye',
      points: [
        'Demo option 6 exports interactive **HTML**.',
        '`pyLDAvis.lda_model.prepare(lda, dtm, vectorizer)`',
        'Explore topic distances and word relevance in a browser.',
        'Exercises return structured Python dicts instead.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Demo - pyLDAvis export',
      subtitle:
        'python module-04-topic-modelling/demo/demo.py - option 6',
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
      title: 'Field rules - Module 4',
      rules: [
        {
          rule: 'Topics are not categories',
          example: 'A case can span multiple themes - weights, not buckets.',
          icon: 'tag',
        },
        {
          rule: 'Inspect top words always',
          example: 'Perplexity guides k; human judgement names the thread.',
          icon: 'eye',
        },
        {
          rule: 'Preprocessing shapes topics',
          example: 'min_df and stopwords change which themes emerge.',
          icon: 'filter',
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
        'Run interactively: **`python start.py`**.',
        'Run tests from the exercise folder: **`pytest test_start.py -v`**.',
        'Solutions in **`solution.py`** - try the exercise first!',
      ],
    },
  },
  {
    type: 'welcome',
    content: {
      title: 'Exercises',
      points: [
        '01 - Archive Themes: LDA batch report with dominant topic per case',
        '02 - Evidence Board: LDA vs NMF comparison + labelled case groupings',
        '03 - Real-World Topics: 20 Newsgroups audit - purity and contingency matrix',
      ],
    },
  },
  {
    type: 'title',
    content: {
      title: 'Module 4 - Complete',
      subtitle: 'Next: Word Embeddings - meaning beyond counts',
      icon: 'check-circle',
    },
  },
];
