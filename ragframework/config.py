"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Configuration for a :class:`~ragframework.pipeline.rag.RAGPipeline`.

    Attributes:
        chunk_size: Target character length used by chunker ``from_config`` helpers.
        chunk_overlap: Overlap used by chunker ``from_config`` helpers.
        top_k: Number of chunks to keep after optional reranking.
        embedding_dim: Optional dimensionality enforced against vectors produced by
            the configured :class:`~ragframework.base.Embedder`. ``None`` disables
            validation for embedders whose dimension is not known ahead of time.
        retrieve_k: Number of candidate chunks to retrieve before reranking. When None,
            the pipeline uses top_k without a reranker and top_k * 4 with one.
        embed_batch_size: Maximum number of texts passed to ``embedder.embed()`` in a
            single call during ingestion. Hosted APIs and local models both have
            practical batch limits; values must be greater than zero.
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    embedding_dim: int | None = None
    retrieve_k: int | None = None
    embed_batch_size: int = 64

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.retrieve_k is not None and self.retrieve_k <= 0:
            raise ValueError("retrieve_k must be positive when set")
        if self.embedding_dim is not None and self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if self.embed_batch_size <= 0:
            raise ValueError("embed_batch_size must be positive")
