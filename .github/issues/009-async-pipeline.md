---
title: "Async Pipeline Support"
labels: ["enhancement", "help wanted"]
---

## Description

Add an `AsyncRAGPipeline` (or `async_ingest` / `async_query` methods on `RAGPipeline`) so the pipeline can be awaited in async web frameworks like FastAPI and Starlette.

## Motivation

Synchronous pipelines block the event loop when used inside async frameworks. Async support is essential for production API services that need to handle concurrent requests without spinning up threads.

## Acceptance criteria

- [ ] `AsyncRAGPipeline` in `ragframework/pipeline/async_rag.py` **or** `async_ingest` / `async_query` methods added to `RAGPipeline`
- [ ] Abstract async ABCs in `ragframework/base.py`: `AsyncEmbedder`, `AsyncRetriever`, `AsyncGenerator` — **or** use `asyncio.to_thread` to wrap sync implementations
- [ ] All existing sync implementations and tests remain unchanged
- [ ] Tests using `pytest-asyncio` — add it to `[dev]` extra in `pyproject.toml`
- [ ] Example: `examples/async_rag.py` showing usage with `asyncio.run()`
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Files to touch

- `ragframework/pipeline/async_rag.py` — new file (or extend `rag.py`)
- `ragframework/base.py` — optional: async ABCs
- `pyproject.toml` — add `pytest-asyncio` to `[dev]`
- `tests/test_pipeline/test_async_rag.py` — new test file
- `examples/async_rag.py` — new example

## Resources

- [asyncio.to_thread docs](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
