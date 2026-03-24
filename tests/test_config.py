"""Tests for RAGConfig."""

import pytest

from ragframework.config import RAGConfig


def test_defaults():
    cfg = RAGConfig()
    assert cfg.chunk_size == 512
    assert cfg.chunk_overlap == 64
    assert cfg.top_k == 5
    assert cfg.embedding_dim == 384


def test_custom_values():
    cfg = RAGConfig(chunk_size=256, chunk_overlap=32, top_k=3, embedding_dim=768)
    assert cfg.chunk_size == 256
    assert cfg.top_k == 3


def test_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        RAGConfig(chunk_size=0)


def test_invalid_overlap_negative():
    with pytest.raises(ValueError, match="chunk_overlap"):
        RAGConfig(chunk_overlap=-1)


def test_overlap_must_be_less_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        RAGConfig(chunk_size=100, chunk_overlap=100)


def test_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        RAGConfig(top_k=0)
