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
        'Persist the model and **analyse where it breaks**.',
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
        '**joblib** persistence and **error analysis**.',
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
      title: 'M1 - Text preprocessing',
      icon: 'scissors',
      points: [
        'Tokenisation (sentence + word), **NLTK** Punkt.',
        'Cleaning: case folding, regex normalisation, Unicode.',
        'Stopword removal, **stemming vs lemmatisation**.',
        'Chaining steps into a reusable **preprocessing pipeline**.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'M2 - Feature extraction',
      icon: 'bar-chart',
      points: [
        'Vector space model and **document-term matrices**.',
        'Bag-of-Words, n-grams, **TF-IDF** weighting.',
        '**Cosine similarity** for document comparison.',
        'sklearn `CountVectorizer` and `TfidfVectorizer`.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'M3 - Text classification',
      icon: 'target',
      points: [
        'Supervised classification: train/test splits, **sklearn Pipeline**.',
        'Naive Bayes, Logistic Regression, Linear SVM.',
        'Accuracy, precision, recall, **F1**, confusion matrix.',
        'Sentiment analysis and spam detection on real data.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'M4 - Topic modelling',
      icon: 'layers',
      points: [
        'Unsupervised theme discovery - no labels needed.',
        '**LDA** and **NMF**: how they differ, when to use each.',
        'Choosing topic count, interpreting and labelling themes.',
        'Optional visualisation with pyLDAvis.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'M5 - Word embeddings',
      icon: 'share-2',
      points: [
        'Distributional semantics - words as dense vectors.',
        '**Word2Vec** (CBOW, Skip-gram) and **GloVe** pre-trained.',
        'Similarity, analogies, embedding arithmetic.',
        'Static vs contextual embeddings - bridge to M7.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'M6 - POS tagging and NER',
      icon: 'tag',
      points: [
        '**spaCy** pipeline: tokenisation → POS → deps → NER.',
        'Dependency parsing for subject-verb-object extraction.',
        'Named entities: persons, orgs, locations, dates.',
        'Custom rules with **EntityRuler**.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'M7 - Transformers',
      icon: 'brain',
      points: [
        'Self-attention and contextual embeddings (BERT/DistilBERT).',
        'HF `pipeline` for classification, NER, **zero-shot**.',
        'Subword tokenisation: BPE and WordPiece.',
        'Optional **fine-tuning** DistilBERT on custom data.',
      ],
    },
  },

  {
    type: 'title',
    content: {
      title: 'Beyond the course',
      subtitle: 'What else is out there',
      icon: 'compass',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'RAG - Retrieval-Augmented Generation',
      icon: 'search',
      points: [
        'Retrieve relevant documents, feed them as **context** to an LLM.',
        'Reduces **hallucination** - the model answers from your data.',
        'Exactly the pattern in **Exercise 02** (TF-IDF retrieval + chat model).',
        'Production RAG swaps TF-IDF for embedding search + a vector DB.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Vector databases',
      icon: 'database',
      points: [
        'Purpose-built stores for **embedding vectors** at scale.',
        'Pinecone, Weaviate, ChromaDB, **pgvector** (Postgres extension).',
        'Semantic search in milliseconds over millions of documents.',
        'Bridge from M5 word embeddings to **production retrieval**.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Prompt engineering and LLM APIs',
      icon: 'message-square',
      points: [
        '**Role prompting**: "You are an expert linguist..." - set persona and constraints.',
        '**Few-shot**: include labelled examples in the prompt to steer output.',
        '**Retrieval-augmented**: inject relevant documents into the prompt (connects to RAG).',
        '**Chain-of-thought**: ask the model to reason step by step before answering.',
        '**Structured output**: constrain responses to JSON schemas.',
        '**Self-consistency**: sample multiple answers, pick the majority - reduces randomness.',
        '**ReAct**: interleave reasoning and actions - the model thinks, calls a tool, observes, then continues.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'MLOps and model lifecycle',
      icon: 'refresh-cw',
      points: [
        'Experiment tracking: **MLflow**, Weights & Biases.',
        'Model registries, versioning, reproducible pipelines.',
        'Deploy as a **Docker container** on AWS SageMaker, GCP Vertex AI, or Azure ML.',
        'Lightweight option: **Hugging Face Inference Endpoints** or **Replicate** - deploy with a config file.',
        'Self-hosted: **FastAPI** + Docker on any VM or Kubernetes cluster.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Evaluation and benchmarks',
      icon: 'clipboard',
      points: [
        'Standard benchmarks: GLUE, SuperGLUE, MMLU, HELM.',
        'Aggregate metrics can **hide** real failure modes.',
        'Build **custom eval sets** for your domain and users.',
        "Connects to the capstone's **error analysis** discipline.",
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Ethics, bias, and safety',
      icon: 'shield',
      points: [
        'Training data encodes **societal biases** - models amplify them.',
        'Fairness audits across demographics, toxicity filters.',
        '**PII detection** in NLP pipelines - data privacy matters.',
        'Responsible deployment: know what your model gets wrong and **for whom**.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Agentic AI and tool use',
      icon: 'wrench',
      points: [
        'LLMs that **act** - call functions, search the web, run code, query databases.',
        '**Tool calls** (function calling) let models invoke APIs with structured arguments.',
        '**MCP** (Model Context Protocol) - open standard for connecting models to tools and data.',
        'Multi-agent systems: specialised agents that collaborate on complex tasks.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Multimodal models',
      icon: 'eye',
      points: [
        'Text is one modality - modern models handle **vision, audio, and code** together.',
        'Vision-language: GPT-4V, LLaVA - describe images, extract text from documents.',
        'Speech-to-text: **Whisper** - transcription and translation in one model.',
        'Same transformer architecture, different input encoders.',
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
      icon: 'file-text',
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
      title: 'More capstone missions',
      subtitle: 'Freeform exercises',
      icon: 'rocket',
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Exercise 02 - Case chatbot',
      icon: 'message-square',
      points: [
        'Load **statements + cold cases** for a case.',
        'Retrieve relevant documents with **TF-IDF similarity** (M2).',
        'Feed context to a **HF text-generation** model.',
        'Run **spaCy NER** on responses to highlight entities (M6).',
        '**Validate** entity names against case data — flag hallucinations.',
      ],
    },
  },
  {
    type: 'standard',
    content: {
      title: 'Exercise 03 - NLP audit dashboard',
      icon: 'bar-chart',
      points: [
        'Corpus stats, **topic discovery** (M4), **sentiment scan** (M7).',
        '**Entity census** with spaCy NER (M6).',
        '**Classification probe** — TF-IDF baseline F1 (M3).',
        'Assemble a **JSON report** and formatted CLI summary.',
        'Stretch: expose a `/audit` **FastAPI** endpoint.',
      ],
    },
  },

  {
    type: 'cards',
    content: {
      title: 'Choose your mission',
      cards: [
        {
          heading: '01 Open case',
          body: 'Structured capstone: dataset, baseline, transformer, persist and compare.',
        },
        {
          heading: '02 Case chatbot',
          body: 'Build a conversational assistant over Inkwell cases with a HF chat model.',
        },
        {
          heading: '03 NLP audit',
          body: 'Run every course technique against a corpus and produce a structured report.',
        },
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
    type: 'title',
    content: {
      title: 'You built something real',
      subtitle: 'From raw text to trained pipeline',
      icon: 'award',
    },
  },
];
