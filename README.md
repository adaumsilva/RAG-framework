# RAG Framework

[![PyPI version](https://img.shields.io/pypi/v/ragframework.svg)](https://pypi.org/project/ragframework/)
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

Optional reranking is supported between retrieval and generation via `Reranker`; `CrossEncoderReranker` uses the existing `[huggingface]` extra.

---

## Installation

```bash
# Core (numpy only)
pip install ragframework

# With PDF support
pip install "ragframework[pdf]"

# With HuggingFace embeddings (local, no API key)
pip install "ragframework[huggingface]"

# With a vector store
pip install "ragframework[faiss]"
pip install "ragframework[chromadb]"

# With the Anthropic generator
pip install "ragframework[anthropic]"

# Everything
pip install "ragframework[all]"
```

Requires Python 3.10 or newer. To install the latest unreleased code from `main`, or to
contribute, see [CONTRIBUTING.md](CONTRIBUTING.md) for the editable-install setup.

---

## Quick Start

```python
from ragframework import RAGPipeline, RAGConfig
from ragframework.document import TextFileLoader
from ragframework.embeddings import RandomEmbedder   # swap for OpenAIEmbedder
from ragframework.retriever import InMemoryRetriever  # swap for FAISSRetriever
from ragframework.generator import EchoGenerator      # swap for OpenAIGenerator

config = RAGConfig(
    chunk_size=512,
    chunk_overlap=64,
    top_k=5,
    embedding_dim=384,
)

pipeline = RAGPipeline.from_config(
    config,
    loader=TextFileLoader(),
    embedder=RandomEmbedder(dim=384),
    retriever=InMemoryRetriever(),
    generator=EchoGenerator(),
)

# Ingest documents
n_chunks = pipeline.ingest_many(["intro.txt", "reference.txt"])
print(f"Indexed {n_chunks} chunks")

# Query, optionally overriding config.top_k for this call
response = pipeline.query("What is this document about?", top_k=3)
print(f"Question: {response.query}")
print(response.answer)
for chunk in response.source_chunks:
    print(f"  Source: {chunk.metadata.get('source')} — {chunk.content[:80]}…")
```

### Loading CSV and JSON Lines

The built-in tabular loaders need no additional dependencies. Each CSV data row
or JSON Lines object becomes a separate document:

```python
from ragframework.document import CSVLoader, JSONLLoader

csv_loader = CSVLoader(
    content_columns=["title", "body"],
    metadata_columns=["url", "date"],
    id_column="id",
)
documents = csv_loader.load("articles.csv")

jsonl_loader = JSONLLoader(content_key="text", metadata_keys=["source"], id_key="id")
documents = jsonl_loader.load("articles.jsonl")
```

Content fields are joined with `separator="\n"`; JSONL also accepts a list of
content keys. Both loaders accept `encoding`, and CSV accepts `delimiter`.
Without an explicit ID field, IDs use the source path hash and zero-based row
index. Metadata contains the file `source`, a reserved zero-based `row_index`,
and only the selected metadata fields. Selecting a metadata field named `source`
replaces the file path with that field's value.

CSV errors identify one-based data rows (excluding the header); malformed JSON
or missing keys identify one-based lines. JSONL requires string content and
string or integer IDs, preserves the types of selected metadata values, and
rejects blank lines and non-object records with `LoaderError`. Empty files
return no documents.

### Loading HTML files and pages

`HTMLLoader` uses Python's standard library and needs no additional dependencies:

```python
from ragframework.document import HTMLLoader

loader = HTMLLoader(timeout=10.0, user_agent="my-rag-app/1.0")
documents = loader.load("saved-page.html")
# The same loader accepts an HTTP(S) URL:
# documents = loader.load("https://example.com/article")
```

Each source produces one document with `source`, `title`, and `format="html"`
metadata. The loader omits scripts, styles, navigation, templates, noscript, and
head text while preserving the first document title separately, excluding SVG
and MathML titles. It collapses whitespace
and separates block elements without breaking inline words or punctuation.
It reads static HTML and does not execute JavaScript. Local files default to
UTF-8 (`encoding` is configurable); HTTP responses use their declared charset
or fall back to that encoding. File, network, and decoding failures raise
`LoaderError`. A page without readable text produces a document with empty content.

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
| High | OpenAI / Anthropic generator | In Progress |
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
3. Enable the Git hooks: `pre-commit install` (Python 3.10 is required for the isolated mypy hook)
4. Write your code and tests
5. Run the checks: `pre-commit run --all-files` and `pytest tests/ -v`
6. Open a pull request

---

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

---

## Acknowledgements

Architecture inspired by [RAG-Anything](https://github.com/HKUDS/RAG-Anything) by HKUDS.
