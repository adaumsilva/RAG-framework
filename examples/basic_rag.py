"""Basic RAG pipeline example.

Demonstrates a full ingest → query cycle using only the built-in
components (no API keys required).

Run:
    python examples/basic_rag.py
"""

from __future__ import annotations

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
from ragframework.pipeline.rag import RAGPipeline
from ragframework.retriever.in_memory import InMemoryRetriever

SAMPLE_TEXT = """\
Retrieval-Augmented Generation (RAG) is a technique that combines information
retrieval with text generation. Rather than relying solely on knowledge baked
into a language model's parameters, RAG retrieves relevant documents from an
external corpus and injects them as context before generating an answer.

This approach offers several advantages: it keeps the knowledge base up-to-date
without retraining, reduces hallucinations by grounding responses in retrieved
evidence, and allows fine-grained control over what sources are consulted.

A typical RAG pipeline consists of four stages:
1. Document ingestion — load and chunk source documents.
2. Embedding — convert chunks to dense vectors.
3. Retrieval — find the chunks most similar to a query.
4. Generation — produce an answer conditioned on the retrieved context.

RAG Framework provides clean abstractions for each stage so you can swap in
your preferred implementations (OpenAI, HuggingFace, ChromaDB, FAISS, …)
without changing the pipeline logic.
"""


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Write a sample document to a temporary file
    # ------------------------------------------------------------------ #
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write(SAMPLE_TEXT)
        tmp_path = f.name

    print(f"Sample document written to: {tmp_path}\n")

    # ------------------------------------------------------------------ #
    # 2. Build the pipeline
    # ------------------------------------------------------------------ #
    pipeline = RAGPipeline(
        loader=TextFileLoader(),
        chunker=FixedSizeChunker(chunk_size=200, chunk_overlap=40),
        # RandomEmbedder produces random vectors — replace with a real
        # embedder (OpenAI, HuggingFace) for meaningful retrieval.
        embedder=RandomEmbedder(dim=64, seed=42),
        retriever=InMemoryRetriever(),
        # EchoGenerator returns the retrieved chunks as the answer.
        # Replace with an LLM-backed generator for real use.
        generator=EchoGenerator(),
        config=RAGConfig(top_k=3),
    )

    # ------------------------------------------------------------------ #
    # 3. Ingest the document
    # ------------------------------------------------------------------ #
    n_chunks = pipeline.ingest(tmp_path)
    print(f"Ingested {n_chunks} chunks.\n")

    # ------------------------------------------------------------------ #
    # 4. Query the pipeline
    # ------------------------------------------------------------------ #
    query = "What are the stages of a RAG pipeline?"
    print(f"Query: {query!r}\n")

    response = pipeline.query(query)

    print("Answer (retrieved context):")
    print("-" * 60)
    print(response.answer)
    print("-" * 60)
    print(f"\nSource chunks used: {len(response.source_chunks)}")

    # Clean up
    Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
