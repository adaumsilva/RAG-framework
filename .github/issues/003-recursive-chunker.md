---
title: "Recursive / Semantic Text Chunker"
labels: ["good first issue", "enhancement"]
---

## Description

Add a `RecursiveChunker` that splits text using a priority list of separators (`\n\n`, `\n`, `. `, ` `) — falling back to the next separator when a chunk is still too long.

## Motivation

The existing `FixedSizeChunker` splits text at arbitrary character boundaries, which can break mid-sentence or mid-paragraph and degrade retrieval quality. A recursive approach preserves semantic boundaries.

## Acceptance criteria

- [ ] Class `RecursiveChunker` in `ragframework/document/chunkers.py`
- [ ] Subclasses `TextChunker` from `ragframework/base.py`
- [ ] Configurable `separators: list[str]`, `chunk_size: int`, `chunk_overlap: int`
- [ ] Falls back through separator list until chunks fit within `chunk_size`
- [ ] Unit tests covering: long word (no separator fits), empty doc, single-separator doc, overlap behaviour
- [ ] Exported from `ragframework/document/__init__.py`
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Files to touch

- `ragframework/document/chunkers.py` — add `RecursiveChunker`
- `ragframework/document/__init__.py` — export it
- `tests/test_document/test_chunkers.py` — add tests

## Resources

- LangChain's `RecursiveCharacterTextSplitter` for inspiration
- Existing chunkers in `ragframework/document/chunkers.py` as reference
