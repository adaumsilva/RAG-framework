---
title: "HuggingFace Sentence Transformers Embedder"
labels: ["good first issue", "enhancement"]
---

## Description

Add a `HuggingFaceEmbedder` using the `sentence-transformers` library for fully local, open-weight embeddings — no API key required.

## Motivation

Not every user wants to call an external API. Local embeddings are faster for bulk ingestion, free, and privacy-preserving. `all-MiniLM-L6-v2` is a solid default that balances quality and speed.

## Acceptance criteria

- [ ] Class `HuggingFaceEmbedder` in `ragframework/embeddings/huggingface.py`
- [ ] Subclasses `Embedder` from `ragframework/base.py`
- [ ] Configurable `model_name` (default: `"all-MiniLM-L6-v2"`) and optional `device` (`"cpu"` / `"cuda"`)
- [ ] Uses `sentence-transformers` (already in `[huggingface]` extra in `pyproject.toml`)
- [ ] Import guarded with helpful message; raises `EmbedderError` on failure
- [ ] Unit tests — mock the model or use a tiny test model to avoid large downloads in CI
- [ ] Exported from `ragframework/embeddings/__init__.py`
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

## Files to touch

- `ragframework/embeddings/huggingface.py` — new file
- `ragframework/embeddings/__init__.py` — export it
- `tests/test_embeddings/test_huggingface.py` — new test file

## Resources

- [Sentence Transformers docs](https://www.sbert.net/)
- [Pretrained model list](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html)
- `RandomEmbedder` in `ragframework/embeddings/random_embedder.py` as reference
