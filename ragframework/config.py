"""Pipeline configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Configuration for a :class:`~ragframework.pipeline.rag.RAGPipeline`.

    Attributes:
        chunk_size: Target character length for each text chunk.
        chunk_overlap: Number of overlapping characters between adjacent chunks.
        top_k: Number of chunks to retrieve per query.
        embedding_dim: Dimensionality of embedding vectors (must match the
            :class:`~ragframework.base.Embedder` in use).
    """

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    embedding_dim: int = 384

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
