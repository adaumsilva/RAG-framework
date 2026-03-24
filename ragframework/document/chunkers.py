"""Built-in text chunkers.

These chunkers have zero external dependencies. For semantic or
recursive chunking strategies, see `.github/GOOD_FIRST_ISSUES.md`.
"""

from __future__ import annotations

from ragframework.base import Chunk, Document, TextChunker
from ragframework.config import RAGConfig


class FixedSizeChunker(TextChunker):
    """Split text into fixed-size character windows with optional overlap.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @classmethod
    def from_config(cls, config: RAGConfig) -> FixedSizeChunker:
        return cls(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        index = 0
        chunk_num = 0
        while index < len(text):
            end = min(index + self.chunk_size, len(text))
            chunk_text = text[index:end]
            chunks.append(
                Chunk(
                    id=f"{document.id}:{chunk_num}",
                    content=chunk_text,
                    metadata={**document.metadata, "chunk_index": chunk_num},
                )
            )
            chunk_num += 1
            index += step
        return chunks


class SentenceChunker(TextChunker):
    """Group sentences into chunks that stay within *max_sentences*.

    Sentences are split naively on ``. ``, ``! ``, and ``? ``.  For
    production-quality sentence segmentation, consider integrating spaCy
    or NLTK (a great contribution opportunity!).

    Args:
        max_sentences: Maximum sentences per chunk.
        overlap_sentences: Number of sentences to repeat at the start of the
            next chunk (context carry-over).
    """

    def __init__(self, max_sentences: int = 5, overlap_sentences: int = 1) -> None:
        if overlap_sentences >= max_sentences:
            raise ValueError("overlap_sentences must be less than max_sentences")
        self.max_sentences = max_sentences
        self.overlap_sentences = overlap_sentences

    def _split_sentences(self, text: str) -> list[str]:
        import re

        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if s.strip()]

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = self._split_sentences(document.content)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        step = self.max_sentences - self.overlap_sentences
        index = 0
        chunk_num = 0
        while index < len(sentences):
            window = sentences[index : index + self.max_sentences]
            content = " ".join(window)
            chunks.append(
                Chunk(
                    id=f"{document.id}:{chunk_num}",
                    content=content,
                    metadata={**document.metadata, "chunk_index": chunk_num},
                )
            )
            chunk_num += 1
            index += step
        return chunks
