# RAG Framework

[![PyPI version](https://img.shields.io/badge/pypi-coming%20soon-lightgrey)](https://github.com/adaumsilva/RAG-framework)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/adaumsilva/RAG-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/adaumsilva/RAG-framework/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/adaumsilva/RAG-framework/branch/main/graph/badge.svg)](https://codecov.io/gh/adaumsilva/RAG-framework)
[![GitHub issues](https://img.shields.io/github/issues/adaumsilva/RAG-framework)](https://github.com/adaumsilva/RAG-framework/issues)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/adaumsilva/RAG-framework/blob/main/CONTRIBUTING.md)
[![GitHub contributors](https://img.shields.io/github/contributors/adaumsilva/RAG-framework)](https://github.com/adaumsilva/RAG-framework/graphs/contributors)
[![GitHub stars](https://img.shields.io/github/stars/adaumsilva/RAG-framework?style=social)](https://github.com/adaumsilva/RAG-framework/stargazers)

A modular, extensible Python framework for building **Retrieval-Augmented Generation (RAG)** pipelines. Plug in your own loaders, embedders, vector stores, and generators — or use the built-in implementations to get started in minutes.

---

## Features

- **Modular by design** — every component (loader, chunker, embedder, retriever, generator) is an abstract base class you can swap out
- **Works out of the box** — built-in text/Markdown loaders, fixed-size chunker, in-memory cosine retriever, and placeholder implementations that need no API keys
- **Extensible ecosystem** — simple contracts mean integrating OpenAI, HuggingFace, ChromaDB, FAISS, or any other tool is just a subclass away
- **Batteries-optional** — core dependency is `numpy` only; add `[pdf]`, `[openai]`, `[chromadb]`, … as you need them
- **Fully tested** — pytest-based test suite with coverage reporting
- **Contributor-friendly** — clear abstractions, good first issues, and detailed contributing guide

---

## Architecture

```
                     ┌─────────────────────────────────────┐
                     │            RAGPipeline               │
                     └─────────────┬───────────────────────┘
                                   │
          ┌────────────────────────┼─────────────────────────┐
          │                        │                         │
          ▼                        ▼                         ▼
  ┌───────────────┐       ┌──────────────┐         ┌──────────────────┐
  │ DocumentLoader│──────▶│  TextChunker │──────┐  │                  │
  └───────────────┘       └──────────────┘      │  │                  │
  (TextFileLoader,        (FixedSizeChunker,     │  │                  │
   MarkdownLoader,         SentenceChunker,      │  │                  │
   PDFLoader*, …)          SemanticChunker*)     │  │                  │
                                                 ▼  │                  │
                                          ┌──────────────┐             │
                                          │   Embedder   │             │
                                          └──────┬───────┘             │
                                                 │  (RandomEmbedder,   │
                                                 │   OpenAIEmbedder*,  │
                                                 │   HFEmbedder*)      │
                                                 ▼                     │
                                          ┌──────────────┐             │
                                          │  Retriever   │             │
                                          └──────┬───────┘             │
                                                 │  (InMemoryRetriever,│
                                                 │   FAISSRetriever*,  │
                                                 │   ChromaRetriever*) │
                                                 ▼                     │
                                          ┌──────────────┐             │
                                          │  Generator   │◀────────────┘
                                          └──────────────┘
                                    (EchoGenerator,
                                     OpenAIGenerator*,
                                     AnthropicGenerator*)

  * = open contribution opportunity — see .github/GOOD_FIRST_ISSUES.md
```

---

## Installation

> **Note:** The package is not yet on PyPI. Install directly from GitHub:

```bash
# Core (numpy only)
pip install git+https://github.com/adaumsilva/RAG-framework.git

# With PDF support
pip install "ragframework[pdf] @ git+https://github.com/adaumsilva/RAG-framework.git"

# With OpenAI support
pip install "ragframework[openai] @ git+https://github.com/adaumsilva/RAG-framework.git"

# With HuggingFace embeddings
pip install "ragframework[huggingface] @ git+https://github.com/adaumsilva/RAG-framework.git"

# Everything
pip install "ragframework[all] @ git+https://github.com/adaumsilva/RAG-framework.git"
```

Once published to PyPI, installation will simplify to `pip install ragframework`.

---

## Quick Start

```python
from ragframework import RAGPipeline, RAGConfig
from ragframework.document import TextFileLoader, FixedSizeChunker
from ragframework.embeddings import RandomEmbedder   # swap for OpenAIEmbedder
from ragframework.retriever import InMemoryRetriever  # swap for FAISSRetriever
from ragframework.generator import EchoGenerator      # swap for OpenAIGenerator

pipeline = RAGPipeline(
    loader=TextFileLoader(),
    chunker=FixedSizeChunker(chunk_size=512, chunk_overlap=64),
    embedder=RandomEmbedder(dim=384),
    retriever=InMemoryRetriever(),
    generator=EchoGenerator(),
    config=RAGConfig(top_k=5),
)

# Ingest a document
n_chunks = pipeline.ingest("my_document.txt")
print(f"Indexed {n_chunks} chunks")

# Query
response = pipeline.query("What is this document about?")
print(response.answer)
for chunk in response.source_chunks:
    print(f"  Source: {chunk.metadata.get('source')} — {chunk.content[:80]}…")
```

### Implementing your own component

```python
from ragframework.base import Embedder

class MyEmbedder(Embedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        # call your embedding API / model here
        ...
```

That's it — plug `MyEmbedder()` into `RAGPipeline` and everything else stays the same.

---

## Roadmap

Community contributions are the engine that drives this roadmap. Pick up a [Good First Issue](.github/GOOD_FIRST_ISSUES.md) and open a PR!

| Priority | Item | Status |
|----------|------|--------|
| High | PDF document loader | Open |
| High | DOCX document loader | Open |
| High | OpenAI embeddings integration | Open |
| High | HuggingFace Sentence Transformers | Open |
| High | OpenAI / Anthropic generator | Open |
| Medium | FAISS vector store retriever | Open |
| Medium | ChromaDB retriever integration | Open |
| Medium | Semantic / recursive chunker | Open |
| Medium | Async pipeline support | Open |
| Low | Jupyter notebook examples | Open |

---

## Contributing

Contributions are what make open source great. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a PR.

1. Fork the repo and create a branch: `git checkout -b feat/my-feature`
2. Install dev dependencies: `pip install -e ".[dev]"`
3. Write your code and tests
4. Run the suite: `pytest tests/ -v`
5. Open a pull request

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

## Acknowledgements

Architecture inspired by [RAG-Anything](https://github.com/HKUDS/RAG-Anything) by HKUDS.
