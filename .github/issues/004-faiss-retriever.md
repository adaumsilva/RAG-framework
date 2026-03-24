---
title: "FAISS Vector Store Retriever"
labels: ["good first issue", "enhancement"]
---

## Description

Add a `FAISSRetriever` backed by Facebook AI Similarity Search for fast approximate nearest-neighbour search.

## Motivation

`InMemoryRetriever` is O(N) at query time and unsuitable for large corpora. FAISS scales to millions of vectors with sub-millisecond query latency via efficient indexing structures.

## Acceptance criteria

- [ ] Class `FAISSRetriever` in `ragframework/retriever/faiss.py`
- [ ] Subclasses `Retriever` from `ragframework/base.py`
- [ ] Uses `faiss-cpu` (already listed in `[faiss]` optional extra in `pyproject.toml`)
- [ ] Import guarded with helpful message pointing to `pip install ragframework[faiss]`
- [ ] Raises `RetrieverError` on failure
- [ ] Unit tests in `tests/test_retriever/test_faiss.py`
- [ ] Exported from `ragframework/retriever/__init__.py`
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Files to touch

- `ragframework/retriever/faiss.py` — new file
- `ragframework/retriever/__init__.py` — export it
- `tests/test_retriever/test_faiss.py` — new test file

## Resources

- [faiss-cpu on PyPI](https://pypi.org/project/faiss-cpu/)
- [FAISS wiki](https://github.com/facebookresearch/faiss/wiki)
- `InMemoryRetriever` in `ragframework/retriever/in_memory.py` as reference
