"""RAG Framework — a modular, extensible Python framework for RAG pipelines."""

from ragframework.base import (
    Chunk,
    Document,
    DocumentLoader,
    Embedder,
    Generator,
    RAGResponse,
    Reranker,
    Retriever,
    TextChunker,
)
from ragframework.config import RAGConfig
from ragframework.pipeline.rag import RAGPipeline

__version__ = "0.2.0"
__all__ = [
    "__version__",
    # Data types
    "Document",
    "Chunk",
    "RAGResponse",
    # Abstract base classes
    "DocumentLoader",
    "TextChunker",
    "Embedder",
    "Retriever",
    "Reranker",
    "Generator",
    # Config
    "RAGConfig",
    # Pipeline
    "RAGPipeline",
]
