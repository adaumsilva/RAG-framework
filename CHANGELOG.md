# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- PDFLoader with support for per-page and whole-file modes (closes #1)
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

[Unreleased]: https://github.com/adaumsilva/RAG-framework/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/adaumsilva/RAG-framework/releases/tag/v0.1.0
