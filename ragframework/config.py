"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Configuration for a :class:`~ragframework.pipeline.rag.RAGPipeline`.

    Attributes:
        chunk_size: Target character length used by chunker ``from_config`` helpers.
        chunk_overlap: Overlap used by chunker ``from_config`` helpers.
        top_k: Number of chunks to retrieve per query.
        embedding_dim: Optional dimensionality enforced against vectors produced by
            the configured :class:`~ragframework.base.Embedder`. ``None`` disables
            validation for embedders whose dimension is not known ahead of time.
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    embedding_dim: int | None = None

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.embedding_dim is not None and self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
