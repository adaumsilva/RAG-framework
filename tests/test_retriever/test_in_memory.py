"""Tests for InMemoryRetriever."""

import pytest

from ragframework.base import Chunk
from ragframework.exceptions import RetrieverError
from ragframework.retriever.in_memory import InMemoryRetriever


def make_chunk(cid: str, embedding: list[float]) -> Chunk:
    return Chunk(id=cid, content=f"Content {cid}", embedding=embedding)


class TestInMemoryRetriever:
    def test_empty_retriever_returns_empty(self):
        r = InMemoryRetriever()
        result = r.retrieve([0.1, 0.2, 0.3], top_k=3)
        assert result == []

    def test_add_and_retrieve(self, embedded_chunks):
        r = InMemoryRetriever()
        r.add(embedded_chunks)
        assert len(r) == len(embedded_chunks)

    def test_retrieve_top_k_limited(self, embedded_chunks):
        r = InMemoryRetriever()
        r.add(embedded_chunks)
        query = embedded_chunks[0].embedding
        results = r.retrieve(query, top_k=2)
        assert len(results) <= 2

    def test_chunk_without_embedding_raises(self):
        r = InMemoryRetriever()
        bad_chunk = Chunk(id="bad", content="no embedding")
        with pytest.raises(RetrieverError, match="no embedding"):
            r.add([bad_chunk])

    def test_retrieve_returns_most_similar_first(self):
        dim = 4
        r = InMemoryRetriever()
        # chunk_a is identical to the query — highest similarity
        chunk_a = make_chunk("a", [1.0, 0.0, 0.0, 0.0])
        chunk_b = make_chunk("b", [0.0, 1.0, 0.0, 0.0])
        chunk_c = make_chunk("c", [0.0, 0.0, 1.0, 0.0])
        r.add([chunk_a, chunk_b, chunk_c])
        results = r.retrieve([1.0, 0.0, 0.0, 0.0], top_k=3)
        assert results[0].id == "a"
