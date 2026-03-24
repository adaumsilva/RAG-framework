"""Custom exception hierarchy for RAG Framework."""


class RAGFrameworkError(Exception):
    """Base exception for all RAG Framework errors."""


class LoaderError(RAGFrameworkError):
    """Raised when a :class:`~ragframework.base.DocumentLoader` fails."""


class ChunkerError(RAGFrameworkError):
    """Raised when a :class:`~ragframework.base.TextChunker` fails."""


class EmbedderError(RAGFrameworkError):
    """Raised when an :class:`~ragframework.base.Embedder` fails."""


class RetrieverError(RAGFrameworkError):
    """Raised when a :class:`~ragframework.base.Retriever` fails."""


class GeneratorError(RAGFrameworkError):
    """Raised when a :class:`~ragframework.base.Generator` fails."""


class PipelineError(RAGFrameworkError):
    """Raised for errors in the :class:`~ragframework.pipeline.rag.RAGPipeline`."""
