---
title: "PDF Document Loader"
labels: ["good first issue", "enhancement"]
---

## Description

Add a `PDFLoader` that reads a PDF file and returns one `Document` per page (or one `Document` for the whole file, configurable).

## Motivation

PDF is the most common document format in enterprise and research settings. Without it, RAG Framework cannot process the majority of real-world document corpora.

## Acceptance criteria

- [ ] Class `PDFLoader` in `ragframework/document/loaders.py`
- [ ] Subclasses `DocumentLoader` from `ragframework/base.py`
- [ ] Uses `pypdf` (already listed in the `[pdf]` optional extra in `pyproject.toml`)
- [ ] Raises `LoaderError` (from `ragframework/exceptions.py`) on failure
- [ ] `pypdf` import guarded with a helpful error message pointing to `pip install ragframework[pdf]`
- [ ] Unit tests in `tests/test_document/test_loaders.py`
- [ ] Exported from `ragframework/document/__init__.py`
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Files to touch

- `ragframework/document/loaders.py` — add `PDFLoader`
- `ragframework/document/__init__.py` — export it
- `tests/test_document/test_loaders.py` — add tests

## Resources

- [pypdf docs](https://pypdf.readthedocs.io/)
- Existing loaders (`TextFileLoader`, `MarkdownLoader`) in `ragframework/document/loaders.py` as reference
