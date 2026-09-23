"""RAG pipeline example using OpenAI.

Requires:
    pip install -e ".[dev,openai]"

Set the OPENAI_API_KEY environment variable before running.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragframework.config import RAGConfig
from ragframework.document.chunkers import FixedSizeChunker
from ragframework.document.loaders import TextFileLoader
from ragframework.embeddings.random_embedder import RandomEmbedder
from ragframework.generator.openai import OpenAIGenerator
from ragframework.pipeline.rag import RAGPipeline
from ragframework.retriever.in_memory import InMemoryRetriever

SAMPLE_TEXT = """\
Retrieval-Augmented Generation (RAG) combines information retrieval
with language generation. A RAG pipeline retrieves relevant document
chunks and provides them to a language model as context.

The retrieved context helps the language model produce answers grounded
in the supplied information.
"""


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set the OPENAI_API_KEY environment variable before running.")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(SAMPLE_TEXT)
        tmp_path = f.name

    try:
        pipeline = RAGPipeline(
            loader=TextFileLoader(),
            chunker=FixedSizeChunker(chunk_size=200, chunk_overlap=40),
            # Replace RandomEmbedder with a real embedding implementation
            # for meaningful retrieval.
            embedder=RandomEmbedder(dim=64, seed=42),
            retriever=InMemoryRetriever(),
            generator=OpenAIGenerator(model="gpt-4o-mini"),
            config=RAGConfig(top_k=3),
        )

        n_chunks = pipeline.ingest(tmp_path)
        print(f"Ingested {n_chunks} chunks.\n")

        query = "What is retrieval-augmented generation?"
        response = pipeline.query(query)

        print(f"Question: {query}\n")
        print("Answer:")
        print("-" * 60)
        print(response.answer)
        print("-" * 60)

    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
