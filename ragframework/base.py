"""Abstract base classes that define the contracts for all RAG components.

Every integration (PDF loaders, OpenAI embeddings, ChromaDB retriever, etc.)
must subclass one of these ABCs and implement the required methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------


@dataclass
class Document:
    """A raw document loaded from a source.

    Attributes:
        id: Unique identifier (typically the source path or a hash).
        content: Full text content of the document.
        metadata: Arbitrary key-value pairs (filename, page number, URL, …).
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A text chunk derived from a :class:`Document`.

    Attributes:
        id: Unique identifier (e.g. ``"<doc_id>:<chunk_index>"``).
        content: Text content of this chunk.
        metadata: Inherited / augmented metadata from the parent document.
        embedding: Optional dense vector produced by an :class:`Embedder`.
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class RAGResponse:
    """The final output of :meth:`RAGPipeline.query`.

    Attributes:
        answer: The generated text answer.
        source_chunks: The retrieved chunks used as context.
    """

    answer: str
    source_chunks: list[Chunk]


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------


class DocumentLoader(ABC):
    """Load documents from a source (file path, URL, database, …)."""

    @abstractmethod
    def load(self, source: str) -> list[Document]:
        """Load one or more :class:`Document` objects from *source*.

        Args:
            source: A path, URL, or identifier pointing to the content.

        Returns:
            A list of :class:`Document` objects. May be a single-element list
            for single-file loaders.
        """


class TextChunker(ABC):
    """Split a :class:`Document` into smaller :class:`Chunk` objects."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Chunk *document* into a list of :class:`Chunk` objects.

        Args:
            document: The document to split.

        Returns:
            An ordered list of chunks covering the full document content.
        """


class Embedder(ABC):
    """Convert text strings into dense embedding vectors."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Produce an embedding for each text in *texts*.

        Args:
            texts: Batch of raw text strings.

        Returns:
            A list of float vectors, one per input text. All vectors must have
            the same dimensionality.
        """


class Retriever(ABC):
    """Index chunks and retrieve the most relevant ones for a query."""

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None:
        """Add *chunks* to the index.

        Chunks must have their :attr:`~Chunk.embedding` set before calling
        this method.

        Args:
            chunks: Chunks to index.
        """

    @abstractmethod
    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Return the *top_k* most relevant chunks for *query_embedding*.

        Args:
            query_embedding: Query vector produced by an :class:`Embedder`.
            top_k: Maximum number of chunks to return.

        Returns:
            Up to *top_k* chunks, ordered by relevance (most relevant first).
        """


class Generator(ABC):
    """Generate a natural-language answer given a query and retrieved context."""

    @abstractmethod
    def generate(self, query: str, context: list[Chunk]) -> str:
        """Generate an answer for *query* using *context* chunks.

        Args:
            query: The user's question.
            context: Retrieved chunks that should inform the answer.

        Returns:
            A string answer.
        """
