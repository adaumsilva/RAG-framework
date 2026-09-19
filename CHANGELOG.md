# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-19

### Added
- Optional reranking stage with `Reranker`, `CrossEncoderReranker`, `NoOpReranker`, and configurable pre-rerank retrieval depth (closes #38)
- `AnthropicGenerator` for grounded answers via the Anthropic Messages API (closes #20)
- `ChromaRetriever` for ephemeral and persistent ChromaDB-backed vector retrieval (closes #5)
- `FAISSRetriever` for approximate cosine-similarity search with an HNSW index (closes #4)
- `HuggingFaceEmbedder` for local embeddings with Sentence Transformers (closes #7)
- `RecursiveChunker` for semantics-preserving text splitting (closes #3)
- `PDFLoader` with support for per-page and whole-file modes (closes #1)
- `AsyncRAGPipeline` for asynchronous RAG ingestion and querying using `asyncio.to_thread()` (closes #9)

### Changed
- License metadata now uses an SPDX expression (`license = "MIT"`) in `pyproject.toml`
- Releases are published to PyPI via GitHub Actions trusted publishing (see `RELEASING.md`)

## [0.1.0] - 2026-03-24

### Added
- Initial project scaffold with modular architecture
- Abstract base classes: `DocumentLoader`, `TextChunker`, `Embedder`, `Retriever`, `Generator`
- Core dataclasses: `Document`, `Chunk`, `RAGResponse`
- `RAGConfig` dataclass for pipeline configuration
- Built-in loaders: `TextFileLoader`, `MarkdownLoader`
- Built-in chunkers: `FixedSizeChunker`, `SentenceChunker`
- Placeholder `RandomEmbedder` for testing without API keys
- `InMemoryRetriever` using cosine similarity (numpy-based)
- `EchoGenerator` placeholder for pipeline testing
- `RAGPipeline` orchestrator with `ingest()` and `query()` methods
- Custom exception hierarchy (`RAGFrameworkError`, `LoaderError`, etc.)
- Unit tests for all core components
- Working `examples/basic_rag.py` end-to-end example
- GitHub Actions CI (Python 3.10 / 3.11 / 3.12)
- Issue templates, PR template, and contributor guide
- 10 curated Good First Issues for new contributors

[Unreleased]: https://github.com/adaumsilva/RAG-framework/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/adaumsilva/RAG-framework/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/adaumsilva/RAG-framework/releases/tag/v0.1.0
