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
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
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
        max_chars: Optional maximum number of characters per chunk. If set,
            chunks close early when this limit is reached, and sentences
            longer than the limit are hard-split.
    """

    def __init__(
        self,
        max_sentences: int = 5,
        overlap_sentences: int = 1,
        max_chars: int | None = None,
    ) -> None:
        if max_sentences <= 0:
            raise ValueError("max_sentences must be positive")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be non-negative")
        if overlap_sentences >= max_sentences:
            raise ValueError("overlap_sentences must be less than max_sentences")
        if max_chars is not None and max_chars <= 0:
            raise ValueError("max_chars must be greater than 0")

        self.max_sentences = max_sentences
        self.overlap_sentences = overlap_sentences
        self.max_chars = max_chars

    def _split_sentences(self, text: str) -> list[str]:
        import re

        raw = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in raw if s.strip()]

    def _split_long_sentence(self, sentence: str) -> list[str]:
        if self.max_chars is None:
            return [sentence]

        chunker = RecursiveChunker(
            separators=[" ", ""],
            chunk_size=self.max_chars,
            chunk_overlap=0,
        )

        document = Document(id="sentence", content=sentence, metadata={})
        return [chunk.content for chunk in chunker.chunk(document)]

    def chunk(self, document: Document) -> list[Chunk]:
        sentences = self._split_sentences(document.content)
        if not sentences:
            return []

        chunks = []
        chunk_num = 0

        # Preserve the original behavior when max_chars is not set.
        if self.max_chars is None:
            step = self.max_sentences - self.overlap_sentences
            for index in range(0, len(sentences), step):
                window = sentences[index : index + self.max_sentences]
                content = " ".join(window)

                chunks.append(
                    Chunk(
                        id=f"{document.id}:{chunk_num}",
                        content=content,
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_num,
                        },
                    )
                )
                chunk_num += 1

            return chunks

        # Character-limited behavior.
        index = 0

        while index < len(sentences):
            window = []
            current_length = 0
            oversized_index = None
            closed_early = False

            for sentence_index in range(
                index,
                min(index + self.max_sentences, len(sentences)),
            ):
                sentence = sentences[sentence_index]
                sentence_length = len(sentence)

                if sentence_length > self.max_chars:
                    oversized_index = sentence_index
                    break

                separator_length = 1 if window else 0

                if window and current_length + separator_length + sentence_length > self.max_chars:
                    closed_early = True
                    break

                window.append(sentence)
                current_length += separator_length + sentence_length

            if window:
                content = " ".join(window)

                chunks.append(
                    Chunk(
                        id=f"{document.id}:{chunk_num}",
                        content=content,
                        metadata={
                            **document.metadata,
                            "chunk_index": chunk_num,
                        },
                    )
                )
                chunk_num += 1

            if oversized_index is not None:
                pieces = self._split_long_sentence(sentences[oversized_index])

                for piece in pieces:
                    chunks.append(
                        Chunk(
                            id=f"{document.id}:{chunk_num}",
                            content=piece,
                            metadata={
                                **document.metadata,
                                "chunk_index": chunk_num,
                            },
                        )
                    )
                    chunk_num += 1

                index = oversized_index + 1

            elif closed_early:
                index = max(index + 1, index + len(window) - self.overlap_sentences)

            else:
                index += self.max_sentences - self.overlap_sentences

        return chunks


class RecursiveChunker(TextChunker):
    """Split text recursively using a hierarchy of separators.

    Args:
        separators: List of separators to use. Defaults to paragraphs, lines, sentences, words, chars.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Maximum number of overlapping characters between adjacent chunks.
    """

    def __init__(
        self,
        separators: list[str] | None = None,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.separators = (
            list(separators) if separators is not None else ["\n\n", "\n", ". ", " ", ""]
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @classmethod
    def from_config(cls, config: RAGConfig) -> RecursiveChunker:
        return cls(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        if not document.content:
            return []

        text_chunks = self._split_text(document.content, self.separators)

        chunks: list[Chunk] = []
        for i, content in enumerate(text_chunks):
            chunks.append(
                Chunk(
                    id=f"{document.id}:{i}",
                    content=content,
                    # We create a new dict for metadata as requested to avoid mutating document metadata
                    metadata={**document.metadata, "chunk_index": i},
                )
            )
        return chunks

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        separator = ""
        next_separators: list[str] = []

        if separators:
            for i, sep in enumerate(separators):
                if sep == "":
                    separator = sep
                    next_separators = separators[i + 1 :]
                    break
                if sep in text:
                    separator = sep
                    next_separators = separators[i + 1 :]
                    break

        if separator == "":
            result = []
            step = self.chunk_size - self.chunk_overlap
            idx = 0
            while idx < len(text):
                end = min(idx + self.chunk_size, len(text))
                result.append(text[idx:end])
                if end == len(text):
                    break
                idx += step
                if step <= 0:
                    break
            return result

        pieces = text.split(separator)
        # Reconstruct separators between pieces
        for i in range(len(pieces) - 1):
            pieces[i] += separator

        return self._merge_pieces(pieces, next_separators)

    def _merge_pieces(self, pieces: list[str], next_separators: list[str]) -> list[str]:
        final_chunks: list[str] = []
        current_chunk_pieces: list[str] = []
        current_length = 0

        for piece in pieces:
            if len(piece) > self.chunk_size:
                if current_chunk_pieces:
                    final_chunks.append("".join(current_chunk_pieces))

                    overlap_pieces: list[str] = []
                    for p in reversed(current_chunk_pieces):
                        next_overlap = [p] + overlap_pieces
                        next_len = sum(len(x) for x in next_overlap)
                        if next_len <= self.chunk_overlap:
                            overlap_pieces = next_overlap
                        else:
                            break

                    piece = "".join(overlap_pieces) + piece
                    current_chunk_pieces = []
                    current_length = 0

                sub_chunks = self._split_text(piece, next_separators)

                if len(sub_chunks) > 1:
                    final_chunks.extend(sub_chunks[:-1])

                if sub_chunks:
                    current_chunk_pieces = [sub_chunks[-1]]
                    current_length = len(sub_chunks[-1])
            else:
                added_length = len(piece)
                if current_length + added_length <= self.chunk_size:
                    current_chunk_pieces.append(piece)
                    current_length += added_length
                else:
                    final_chunks.append("".join(current_chunk_pieces))

                    overlap_pieces = []
                    for p in reversed(current_chunk_pieces):
                        next_overlap_pieces = [p] + overlap_pieces
                        next_overlap_length = sum(len(x) for x in next_overlap_pieces)

                        if (
                            next_overlap_length <= self.chunk_overlap
                            and next_overlap_length + len(piece) <= self.chunk_size
                        ):
                            overlap_pieces = next_overlap_pieces
                        else:
                            break

                    current_chunk_pieces = overlap_pieces + [piece] if overlap_pieces else [piece]
                    current_length = sum(len(x) for x in current_chunk_pieces)

        if current_chunk_pieces:
            final_chunks.append("".join(current_chunk_pieces))

        return final_chunks
