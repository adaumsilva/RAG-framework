---
title: "ChromaDB Retriever Integration"
labels: ["good first issue", "enhancement"]
---

## Description

Add a `ChromaRetriever` backed by ChromaDB, a popular open-source vector database with persistent storage and metadata filtering.

## Motivation

ChromaDB provides on-disk persistence, rich metadata filtering, and a simple Python API — ideal for prototype-to-production workflows where data must survive process restarts.

## Acceptance criteria

- [ ] Class `ChromaRetriever` in `ragframework/retriever/chroma.py`
- [ ] Subclasses `Retriever` from `ragframework/base.py`
- [ ] Uses `chromadb` (already listed in `[chromadb]` optional extra in `pyproject.toml`)
- [ ] Supports both ephemeral (`client = chromadb.Client()`) and persistent (`chromadb.PersistentClient(path=...)`) modes via constructor argument
- [ ] Import guarded; raises `RetrieverError` on failure
- [ ] Unit tests using ephemeral mode (no disk I/O)
- [ ] Exported from `ragframework/retriever/__init__.py`
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Files to touch

- `ragframework/retriever/chroma.py` — new file
- `ragframework/retriever/__init__.py` — export it
- `tests/test_retriever/test_chroma.py` — new test file

## Resources

- [ChromaDB docs](https://docs.trychroma.com/)
- `InMemoryRetriever` in `ragframework/retriever/in_memory.py` as reference
