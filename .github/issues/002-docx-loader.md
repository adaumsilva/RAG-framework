---
title: "DOCX Document Loader"
labels: ["good first issue", "enhancement"]
---

## Description

Add a `DocxLoader` that reads `.docx` files using `python-docx` and returns a `Document` per file (concatenating all paragraphs).

## Motivation

Word documents are ubiquitous in business workflows. Supporting DOCX opens the framework to HR, legal, and operational use-cases.

## Acceptance criteria

- [ ] Class `DocxLoader` in `ragframework/document/loaders.py`
- [ ] Subclasses `DocumentLoader` from `ragframework/base.py`
- [ ] Uses `python-docx` — add it to the `[docx]` extra in `pyproject.toml`
- [ ] Raises `LoaderError` on failure; import guarded with helpful message
- [ ] Unit tests in `tests/test_document/test_loaders.py`
- [ ] Exported from `ragframework/document/__init__.py`
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Files to touch

- `ragframework/document/loaders.py` — add `DocxLoader`
- `ragframework/document/__init__.py` — export it
- `pyproject.toml` — add `python-docx` to `[docx]` extra
- `tests/test_document/test_loaders.py` — add tests

## Resources

- [python-docx docs](https://python-docx.readthedocs.io/)
- Existing loaders (`TextFileLoader`, `MarkdownLoader`) in `ragframework/document/loaders.py` as reference
