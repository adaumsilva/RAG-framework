# Good First Issues

A curated backlog of well-scoped issues for new contributors.
Each entry below maps 1-to-1 with a GitHub issue that will be opened with the `good first issue` label.

Pick one, comment on the issue to claim it, and open a PR!

---

## Issue #1 — PDF Document Loader

**Labels:** `good first issue`, `enhancement`
**Effort:** Small (< 2 hours)

### Description
Add a `PDFLoader` that reads a PDF file and returns one `Document` per page (or one `Document` for the whole file, configurable).

### Motivation
PDF is the most common document format in enterprise and research settings. Without it, RAG Framework cannot process the majority of real-world document corpora.

### Acceptance criteria
- [ ] Class `PDFLoader` in `ragframework/document/loaders.py`
- [ ] Subclasses `DocumentLoader` from `ragframework/base.py`
- [ ] Uses `pypdf` (already in the `[pdf]` optional extra in `pyproject.toml`)
- [ ] Raises `LoaderError` (from `ragframework/exceptions.py`) on failure
- [ ] `pypdf` import guarded with a helpful error message
- [ ] Unit tests in `tests/test_document/test_loaders.py`
- [ ] Exported from `ragframework/document/__init__.py`
- [ ] `CHANGELOG.md` updated

### Files to touch
- `ragframework/document/loaders.py` — add `PDFLoader`
- `ragframework/document/__init__.py` — export it
- `tests/test_document/test_loaders.py` — add tests

### Resources
- [pypdf docs](https://pypdf.readthedocs.io/)

---

## Issue #2 — DOCX Document Loader

**Labels:** `good first issue`, `enhancement`
**Effort:** Small (< 2 hours)

### Description
Add a `DocxLoader` that reads `.docx` files and returns a `Document` per paragraph or per file.

### Motivation
Word documents are ubiquitous in business workflows. Supporting DOCX opens the framework to HR, legal, and operational use-cases.

### Acceptance criteria
- [ ] Class `DocxLoader` in `ragframework/document/loaders.py`
- [ ] Subclasses `DocumentLoader`
- [ ] Uses `python-docx` (add to `[docx]` extra in `pyproject.toml`)
- [ ] Raises `LoaderError` on failure; import guarded
- [ ] Unit tests
- [ ] Exported from `ragframework/document/__init__.py`
- [ ] `CHANGELOG.md` updated

### Resources
- [python-docx docs](https://python-docx.readthedocs.io/)

---

## Issue #3 — Semantic / Recursive Text Chunker

**Labels:** `good first issue`, `enhancement`
**Effort:** Medium (half day)

### Description
Add a `RecursiveChunker` that splits text using a priority list of separators (`\n\n`, `\n`, `. `, ` `) — falling back to the next separator when a chunk is still too long. Inspired by LangChain's `RecursiveCharacterTextSplitter`.

### Motivation
The `FixedSizeChunker` may split mid-sentence or mid-paragraph, degrading retrieval quality. A recursive approach preserves semantic boundaries.

### Acceptance criteria
- [ ] Class `RecursiveChunker` in `ragframework/document/chunkers.py`
- [ ] Subclasses `TextChunker`
- [ ] Configurable `separators`, `chunk_size`, `chunk_overlap`
- [ ] Unit tests covering edge cases (long word, empty doc, single-separator doc)
- [ ] Exported from `ragframework/document/__init__.py`
- [ ] `CHANGELOG.md` updated

---

## Issue #4 — FAISS Vector Store Retriever

**Labels:** `good first issue`, `enhancement`
**Effort:** Medium (half day)

### Description
Add a `FAISSRetriever` backed by Facebook AI Similarity Search for fast approximate nearest-neighbour search.

### Motivation
`InMemoryRetriever` is O(N) and unsuitable for large corpora. FAISS scales to millions of vectors with sub-millisecond query latency.

### Acceptance criteria
- [ ] Class `FAISSRetriever` in `ragframework/retriever/faiss.py`
- [ ] Subclasses `Retriever`
- [ ] Uses `faiss-cpu` (already listed in `[faiss]` extra)
- [ ] Import guarded with helpful message
- [ ] Raises `RetrieverError` on failure
- [ ] Unit tests in `tests/test_retriever/test_faiss.py`
- [ ] Exported from `ragframework/retriever/__init__.py`
- [ ] `CHANGELOG.md` updated

### Resources
- [faiss-cpu PyPI](https://pypi.org/project/faiss-cpu/)
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki)

---

## Issue #5 — ChromaDB Retriever Integration

**Labels:** `good first issue`, `enhancement`
**Effort:** Medium (half day)

### Description
Add a `ChromaRetriever` backed by ChromaDB, a popular open-source vector database.

### Motivation
ChromaDB provides persistent storage, metadata filtering, and a simple API — ideal for prototype-to-production workflows.

### Acceptance criteria
- [ ] Class `ChromaRetriever` in `ragframework/retriever/chroma.py`
- [ ] Subclasses `Retriever`
- [ ] Uses `chromadb` (already in `[chromadb]` extra)
- [ ] Supports both ephemeral (in-memory) and persistent (on-disk) modes
- [ ] Import guarded; raises `RetrieverError` on failure
- [ ] Unit tests (using ephemeral mode)
- [ ] Exported from `ragframework/retriever/__init__.py`
- [ ] `CHANGELOG.md` updated

### Resources
- [ChromaDB docs](https://docs.trychroma.com/)

---

## Issue #6 — OpenAI Embeddings Integration

**Labels:** `good first issue`, `enhancement`
**Effort:** Small (< 2 hours)

### Description
Add an `OpenAIEmbedder` that calls the OpenAI Embeddings API (`text-embedding-3-small` by default).

### Motivation
OpenAI embeddings are high-quality, widely used, and the most common starting point for RAG applications.

### Acceptance criteria
- [ ] Class `OpenAIEmbedder` in `ragframework/embeddings/openai.py`
- [ ] Subclasses `Embedder`
- [ ] Configurable `model` and `api_key` (falls back to `OPENAI_API_KEY` env var)
- [ ] Uses `openai` SDK (already in `[openai]` extra)
- [ ] Import guarded; raises `EmbedderError` on failure
- [ ] Unit tests using `unittest.mock` to avoid real API calls
- [ ] Exported from `ragframework/embeddings/__init__.py`
- [ ] Usage example added to `examples/` or README
- [ ] `CHANGELOG.md` updated

### Resources
- [OpenAI Embeddings docs](https://platform.openai.com/docs/guides/embeddings)

---

## Issue #7 — HuggingFace Sentence Transformers Embedder

**Labels:** `good first issue`, `enhancement`
**Effort:** Small (< 2 hours)

### Description
Add a `HuggingFaceEmbedder` using the `sentence-transformers` library for fully local, open-weight embeddings.

### Motivation
Not every user wants to call an external API. Local embeddings are faster for bulk ingestion, free, and privacy-preserving.

### Acceptance criteria
- [ ] Class `HuggingFaceEmbedder` in `ragframework/embeddings/huggingface.py`
- [ ] Subclasses `Embedder`
- [ ] Configurable `model_name` (default: `"all-MiniLM-L6-v2"`)
- [ ] Uses `sentence-transformers` (already in `[huggingface]` extra)
- [ ] Import guarded; raises `EmbedderError` on failure
- [ ] Unit tests using a tiny model or mock
- [ ] Exported from `ragframework/embeddings/__init__.py`
- [ ] `CHANGELOG.md` updated

### Resources
- [Sentence Transformers docs](https://www.sbert.net/)

---

## Issue #8 — OpenAI / Anthropic Generator

**Labels:** `good first issue`, `enhancement`
**Effort:** Small (< 2 hours)

### Description
Add `OpenAIGenerator` and/or `AnthropicGenerator` that format retrieved chunks as context and call the respective chat completion API.

### Motivation
`EchoGenerator` is a placeholder. Real applications need an LLM to synthesise an answer from the retrieved context.

### Acceptance criteria
- [ ] `OpenAIGenerator` in `ragframework/generator/openai.py` (and/or `AnthropicGenerator` in `ragframework/generator/anthropic.py`)
- [ ] Subclasses `Generator`
- [ ] Configurable `model`, `api_key`, `system_prompt`, `max_tokens`
- [ ] Formats context chunks into a readable prompt
- [ ] Import guarded; raises `GeneratorError` on failure
- [ ] Unit tests using mocks
- [ ] Exported from `ragframework/generator/__init__.py`
- [ ] Usage example in `examples/`
- [ ] `CHANGELOG.md` updated

### Resources
- [OpenAI Chat Completions](https://platform.openai.com/docs/guides/text-generation)
- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)

---

## Issue #9 — Async Pipeline Support

**Labels:** `enhancement`, `help wanted`
**Effort:** Large (1-2 days)

### Description
Add an `AsyncRAGPipeline` (or async methods on `RAGPipeline`) so that `ingest` and `query` can be awaited, enabling concurrent document ingestion and serving async web applications.

### Motivation
Synchronous pipelines block the event loop in async frameworks (FastAPI, Starlette). Async support is essential for production API services.

### Acceptance criteria
- [ ] `AsyncRAGPipeline` in `ragframework/pipeline/async_rag.py` (or `async_ingest` / `async_query` methods on `RAGPipeline`)
- [ ] Abstract async counterparts added to `ragframework/base.py` (`AsyncEmbedder`, `AsyncRetriever`, `AsyncGenerator`) — or use `asyncio.to_thread` for sync wrappers
- [ ] All existing sync implementations remain unchanged
- [ ] Tests using `pytest-asyncio`
- [ ] Example: `examples/async_rag.py`
- [ ] `CHANGELOG.md` updated

---

## Issue #10 — Jupyter Notebook Example & Expanded Docs

**Labels:** `good first issue`, `documentation`
**Effort:** Small (< 2 hours)

### Description
Add a Jupyter notebook (`examples/basic_rag.ipynb`) that walks through the full pipeline step-by-step with prose explanations, and expand the `docs/` directory with a getting-started guide.

### Motivation
A notebook lowers the barrier to entry for data scientists and ML practitioners who prefer interactive environments over scripts.

### Acceptance criteria
- [ ] `examples/basic_rag.ipynb` that runs end-to-end using built-in components
- [ ] Notebook cells include markdown explanations for each pipeline stage
- [ ] `docs/getting-started.md` with installation, quickstart, and links to the ABCs
- [ ] All notebook cells execute without errors (`jupyter nbconvert --to notebook --execute`)
- [ ] `CHANGELOG.md` updated
