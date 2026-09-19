"""Tests for rerankers."""

import sys
import types

from ragframework.base import Chunk
from ragframework.reranker import CrossEncoderReranker, NoOpReranker


class FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        return [0.1, 0.9]


def test_cross_encoder_reranks(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    chunks = [Chunk(id="a", content="a"), Chunk(id="b", content="b")]
    result = CrossEncoderReranker().rerank("q", chunks, 1)

    assert result[0].id == "b"
    assert result[0].metadata["rerank_score"] == 0.9


def test_noop_reranker():
    chunks = [Chunk(id="a", content="a"), Chunk(id="b", content="b")]
    assert NoOpReranker().rerank("q", chunks, 1) == chunks[:1]
