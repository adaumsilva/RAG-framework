"""Tests for built-in text chunkers."""

import pytest

from ragframework.base import Document
from ragframework.document.chunkers import FixedSizeChunker, SentenceChunker


@pytest.fixture()
def long_doc():
    return Document(id="d1", content="A" * 200, metadata={})


@pytest.fixture()
def sentence_doc():
    return Document(
        id="d2",
        content="The cat sat on the mat. The dog ran in the park. The bird flew over the hill. "
        "The fish swam in the sea. The ant carried a crumb.",
        metadata={},
    )


class TestFixedSizeChunker:
    def test_produces_chunks(self, long_doc):
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(long_doc)
        assert len(chunks) > 1

    def test_chunk_ids_are_unique(self, long_doc):
        chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(long_doc)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_overlap_not_exceed_size(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=10, chunk_overlap=10)

    def test_short_doc_single_chunk(self):
        doc = Document(id="x", content="Short text.", metadata={})
        chunker = FixedSizeChunker(chunk_size=512, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Short text."

    def test_empty_doc_returns_no_chunks(self):
        doc = Document(id="x", content="", metadata={})
        chunker = FixedSizeChunker()
        chunks = chunker.chunk(doc)
        assert chunks == []


class TestSentenceChunker:
    def test_produces_chunks(self, sentence_doc):
        chunker = SentenceChunker(max_sentences=2, overlap_sentences=0)
        chunks = chunker.chunk(sentence_doc)
        assert len(chunks) >= 2

    def test_overlap_less_than_max_required(self):
        with pytest.raises(ValueError):
            SentenceChunker(max_sentences=2, overlap_sentences=2)

    def test_empty_doc(self):
        doc = Document(id="x", content="", metadata={})
        chunker = SentenceChunker()
        assert chunker.chunk(doc) == []
