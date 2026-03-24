---
title: "OpenAI Embeddings Integration"
labels: ["good first issue", "enhancement"]
---

## Description

Add an `OpenAIEmbedder` that calls the OpenAI Embeddings API (`text-embedding-3-small` by default).

## Motivation

OpenAI embeddings are high-quality, widely used, and the most common starting point for RAG applications. This is the first "real" embedder that replaces the `RandomEmbedder` placeholder.

## Acceptance criteria

- [ ] Class `OpenAIEmbedder` in `ragframework/embeddings/openai.py`
- [ ] Subclasses `Embedder` from `ragframework/base.py`
- [ ] Configurable `model` (default: `"text-embedding-3-small"`) and `api_key` (falls back to `OPENAI_API_KEY` env var)
- [ ] Batches requests efficiently (respects OpenAI's max tokens per request)
- [ ] Uses `openai` SDK (already in `[openai]` extra in `pyproject.toml`)
- [ ] Import guarded with helpful message; raises `EmbedderError` on failure
- [ ] Unit tests using `unittest.mock` to avoid real API calls
- [ ] Exported from `ragframework/embeddings/__init__.py`
- [ ] Usage snippet added to `examples/` or README
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Files to touch

- `ragframework/embeddings/openai.py` — new file
- `ragframework/embeddings/__init__.py` — export it
- `tests/test_embeddings/test_openai.py` — new test file (use mocks)

## Resources

- [OpenAI Embeddings guide](https://platform.openai.com/docs/guides/embeddings)
- `RandomEmbedder` in `ragframework/embeddings/random_embedder.py` as reference
