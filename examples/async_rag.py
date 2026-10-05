from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragframework.config import RAGConfig
from ragframework.document.chunkers import FixedSizeChunker
from ragframework.document.loaders import TextFileLoader
from ragframework.embeddings.random_embedder import RandomEmbedder
from ragframework.generator.echo_generator import EchoGenerator
from ragframework.pipeline.async_rag import AsyncRAGPipeline
from ragframework.retriever.in_memory import InMemoryRetriever

SAMPLE_TEXT = """\
Retrieval-Augmented Generation (RAG) combines information retrieval with
text generation. A typical RAG pipeline loads documents, splits them into
chunks, creates embeddings, retrieves relevant chunks, and generates an
answer from the retrieved context.

AsyncRAGPipeline provides an asynchronous interface for the same pipeline,
making it suitable for async applications such as FastAPI and Starlette.
"""


async def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Write a sample document to a temporary file
    # ------------------------------------------------------------------ #
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(SAMPLE_TEXT)
        tmp_path = f.name

    print(f"Sample document written to: {tmp_path}\n")

    # ------------------------------------------------------------------ #
    # 2. Build the async pipeline
    # ------------------------------------------------------------------ #
    async_pipeline = AsyncRAGPipeline(
        loader=TextFileLoader(),
        chunker=FixedSizeChunker(chunk_size=200, chunk_overlap=40),
        embedder=RandomEmbedder(dim=64, seed=42),
        retriever=InMemoryRetriever(),
        generator=EchoGenerator(),
        config=RAGConfig(top_k=3),
    )

    # ------------------------------------------------------------------ #
    # 3. Asynchronously ingest the document
    # ------------------------------------------------------------------ #
    n_chunks = await async_pipeline.async_ingest(tmp_path)
    print(f"Ingested {n_chunks} chunks.\n")

    # ------------------------------------------------------------------ #
    # 4. Asynchronously query the pipeline
    # ------------------------------------------------------------------ #
    query = "What are the stages of a RAG pipeline?"
    print(f"Query: {query!r}\n")

    response = await async_pipeline.async_query(query)

    print("Answer (retrieved context):")
    print("-" * 60)
    print(response.answer)
    print("-" * 60)
    print(f"\nSource chunks used: {len(response.source_chunks)}")

    # Clean up
    Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
