"""Main RAG pipeline orchestrator."""

from __future__ import annotations

from collections.abc import Iterable

from ragframework.base import (
    Chunk,
    DocumentLoader,
    Embedder,
    Generator,
    RAGResponse,
    Retriever,
    TextChunker,
)
from ragframework.config import RAGConfig
from ragframework.exceptions import PipelineError


class RAGPipeline:
    """Orchestrates the full RAG workflow: load → chunk → embed → index → query.

    Example::

        from ragframework.pipeline.rag import RAGPipeline
        from ragframework.document import TextFileLoader, FixedSizeChunker
        from ragframework.embeddings import RandomEmbedder
        from ragframework.retriever import InMemoryRetriever
        from ragframework.generator import EchoGenerator
        from ragframework.config import RAGConfig

        pipeline = RAGPipeline(
            loader=TextFileLoader(),
            chunker=FixedSizeChunker(),
            embedder=RandomEmbedder(),
            retriever=InMemoryRetriever(),
            generator=EchoGenerator(),
            config=RAGConfig(),
        )

        pipeline.ingest("my_document.txt")
        response = pipeline.query("What is this document about?")
        print(response.answer)

    Args:
        loader: Converts a source path/URL into :class:`~ragframework.base.Document` objects.
        chunker: Splits documents into :class:`~ragframework.base.Chunk` objects.
        embedder: Converts chunk text to dense vectors.
        retriever: Indexes and searches chunks by vector similarity.
        generator: Produces a final answer given the query and retrieved chunks.
        config: Pipeline configuration (chunk sizes, top-k, …).
    """

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: TextChunker,
        embedder: Embedder,
        retriever: Retriever,
        generator: Generator,
        config: RAGConfig | None = None,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.retriever = retriever
        self.generator = generator
        self.config = config or RAGConfig()

    def ingest(self, source: str) -> int:
        """Load, chunk, embed, and index a document.

        Args:
            source: Path, URL, or identifier passed to the configured loader.

        Returns:
            The number of chunks added to the retriever.

        Raises:
            :class:`~ragframework.exceptions.PipelineError`: If any stage fails.
        """
        try:
            documents = self.loader.load(source)
        except Exception as exc:
            raise PipelineError(f"Loading failed for '{source}': {exc}") from exc

        all_chunks: list[Chunk] = []
        for doc in documents:
            try:
                chunks = self.chunker.chunk(doc)
            except Exception as exc:
                raise PipelineError(f"Chunking failed for document '{doc.id}': {exc}") from exc
            all_chunks.extend(chunks)

        if not all_chunks:
            return 0

        texts = [c.content for c in all_chunks]
        try:
            embeddings = self.embedder.embed(texts)
        except Exception as exc:
            raise PipelineError(f"Embedding failed: {exc}") from exc

        for chunk, emb in zip(all_chunks, embeddings, strict=True):
            chunk.embedding = emb

        try:
            self.retriever.add(all_chunks)
        except Exception as exc:
            raise PipelineError(f"Indexing failed: {exc}") from exc

        return len(all_chunks)

    def ingest_many(self, sources: Iterable[str]) -> int:
        """Ingest multiple sources and return the total chunk count.

        Each source is passed through :meth:`ingest` in order. If ingestion of a
        source fails, the raised :class:`~ragframework.exceptions.PipelineError`
        identifies that source and reports how many earlier sources and chunks
        were successfully indexed.

        Args:
            sources: Paths, URLs, or identifiers passed to the configured loader.

        Returns:
            The total number of chunks added to the retriever.

        Raises:
            :class:`~ragframework.exceptions.PipelineError`: If any source fails.
        """
        total_chunks = 0

        for successful_sources, source in enumerate(sources):
            try:
                total_chunks += self.ingest(source)
            except PipelineError as exc:
                source_label = "source" if successful_sources == 1 else "sources"
                chunk_label = "chunk" if total_chunks == 1 else "chunks"
                raise PipelineError(
                    f"Ingest failed for '{source}' after {successful_sources} "
                    f"{source_label} and {total_chunks} {chunk_label} succeeded: {exc}"
                ) from exc

        return total_chunks

    def query(self, query: str, *, top_k: int | None = None) -> RAGResponse:
        """Retrieve relevant chunks and generate an answer.

        Args:
            query: The user's question.
            top_k: Optional per-call override for the number of chunks to retrieve.

        Returns:
            A :class:`~ragframework.base.RAGResponse` with the answer and
            the source chunks used.

        Raises:
            :class:`~ragframework.exceptions.PipelineError`: If any stage fails.
        """
        try:
            query_embedding = self.embedder.embed([query])[0]
        except Exception as exc:
            raise PipelineError(f"Query embedding failed: {exc}") from exc

        retrieve_top_k = self.config.top_k if top_k is None else top_k
        if retrieve_top_k <= 0:
            raise PipelineError("top_k must be positive")

        try:
            chunks = self.retriever.retrieve(query_embedding, top_k=retrieve_top_k)
        except Exception as exc:
            raise PipelineError(f"Retrieval failed: {exc}") from exc

        try:
            answer = self.generator.generate(query, chunks)
        except Exception as exc:
            raise PipelineError(f"Generation failed: {exc}") from exc

        return RAGResponse(answer=answer, source_chunks=chunks, query=query)
