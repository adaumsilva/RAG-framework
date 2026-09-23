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

    def test_max_chars_none_preserves_behavior(self, sentence_doc):
        chunker = SentenceChunker(
            max_sentences=2,
            overlap_sentences=0,
            max_chars=None,
        )

        chunks = chunker.chunk(sentence_doc)

        assert len(chunks) >= 2

    def test_long_sentence_is_split(self):
        doc = Document(
            id="long",
            content="A" * 50,
            metadata={},
        )

        chunker = SentenceChunker(
            max_sentences=5,
            overlap_sentences=0,
            max_chars=10,
        )

        chunks = chunker.chunk(doc)

        assert len(chunks) == 5
        assert all(len(chunk.content) <= 10 for chunk in chunks)
        assert "".join(chunk.content for chunk in chunks) == "A" * 50

    def test_max_chars_closes_window_early(self, sentence_doc):
        chunker = SentenceChunker(
            max_sentences=5,
            overlap_sentences=0,
            max_chars=20,
        )

        chunks = chunker.chunk(sentence_doc)

        assert all(len(chunk.content) <= 20 for chunk in chunks)

    def test_max_chars_preserves_overlap_when_closing_early(self):
        doc = Document(
            id="overlap",
            content="Aaa. Bbb. Ccc. Ddd.",
            metadata={},
        )

        chunker = SentenceChunker(
            max_sentences=5,
            overlap_sentences=1,
            max_chars=9,
        )

        chunks = chunker.chunk(doc)

        assert [chunk.content for chunk in chunks] == [
            "Aaa. Bbb.",
            "Bbb. Ccc.",
            "Ccc. Ddd.",
        ]


class TestRecursiveChunker:
    def test_empty_doc(self):
        from ragframework.document.chunkers import RecursiveChunker

        doc = Document(id="x", content="", metadata={})
        chunker = RecursiveChunker()
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_short_doc(self):
        from ragframework.document.chunkers import RecursiveChunker

        doc = Document(id="x", content="Short", metadata={})
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Short"

    def test_exact_size_doc(self):
        from ragframework.document.chunkers import RecursiveChunker

        doc = Document(id="x", content="Exactly10!", metadata={})
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "Exactly10!"

    def test_oversized_doc_long_word(self, long_doc):
        from ragframework.document.chunkers import RecursiveChunker

        # long_doc is "A" * 200
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(long_doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 50

    def test_single_custom_separator(self):
        from ragframework.document.chunkers import RecursiveChunker

        doc = Document(id="x", content="a,b,c,d,e,f", metadata={})
        chunker = RecursiveChunker(separators=[","], chunk_size=4, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        # "a,b" is 3 chars. "c,d" is 3 chars.
        assert [c.content for c in chunks] == ["a,b,", "c,d,", "e,f"]

    def test_overlap_behavior(self):
        from ragframework.document.chunkers import RecursiveChunker

        # 3 words of length 5 + spaces
        doc = Document(id="x", content="Hello world today", metadata={})
        chunker = RecursiveChunker(chunk_size=12, chunk_overlap=6)
        chunks = chunker.chunk(doc)
        # "Hello world" is 11 chars. Next is "world today" which is 11 chars.
        assert [c.content for c in chunks] == ["Hello world ", "world today"]

    def test_text_preservation(self):
        from ragframework.document.chunkers import RecursiveChunker

        text = "Paragraph 1\n\nParagraph 2 is slightly longer.\n\nParagraph 3."
        doc = Document(id="x", content=text, metadata={})
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        assert "".join(c.content for c in chunks) == text

    def test_metadata_and_ids(self):
        from ragframework.document.chunkers import RecursiveChunker

        doc = Document(id="doc1", content="A B C", metadata={"author": "test"})
        chunker = RecursiveChunker(chunk_size=1, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        assert chunks[0].id == "doc1:0"
        assert chunks[0].metadata["author"] == "test"
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[1].id == "doc1:1"

    def test_validation(self):
        from ragframework.document.chunkers import RecursiveChunker

        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=0)
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=10, chunk_overlap=-1)
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=10, chunk_overlap=10)
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=10, chunk_overlap=15)

    def test_edge_cases(self):
        from ragframework.document.chunkers import RecursiveChunker

        # empty separators
        doc = Document(id="x", content="abcdef", metadata={})
        chunker = RecursiveChunker(separators=[], chunk_size=3, chunk_overlap=1)
        chunks = chunker.chunk(doc)
        assert [c.content for c in chunks] == ["abc", "cde", "ef"]

        # duplicate separators
        chunker = RecursiveChunker(separators=[" ", " "], chunk_size=4, chunk_overlap=0)
        doc2 = Document(id="x", content="a b c", metadata={})
        chunks = chunker.chunk(doc2)
        assert [c.content for c in chunks] == ["a b ", "c"]

    def test_redundant_trailing_chunks(self):
        from ragframework.document.chunkers import RecursiveChunker

        doc = Document(id="x", content="x" * 10, metadata={})
        chunker = RecursiveChunker(
            separators=[],
            chunk_size=6,
            chunk_overlap=4,
        )
        chunks = chunker.chunk(doc)
        assert [len(c.content) for c in chunks] == [6, 6, 6]
