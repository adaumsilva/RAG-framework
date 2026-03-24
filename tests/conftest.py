"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from ragframework.base import Chunk, Document
from ragframework.config import RAGConfig
from ragframework.document.chunkers import FixedSizeChunker
from ragframework.document.loaders import TextFileLoader
from ragframework.embeddings.random_embedder import RandomEmbedder
from ragframework.generator.echo_generator import EchoGenerator
from ragframework.retriever.in_memory import InMemoryRetriever


@pytest.fixture()
def sample_document() -> Document:
    return Document(
        id="doc-001",
        content="The quick brown fox jumps over the lazy dog. " * 20,
        metadata={"source": "test"},
    )


@pytest.fixture()
def sample_chunks(sample_document: Document) -> list[Chunk]:
    chunker = FixedSizeChunker(chunk_size=50, chunk_overlap=10)
    return chunker.chunk(sample_document)


@pytest.fixture()
def embedded_chunks(sample_chunks: list[Chunk]) -> list[Chunk]:
    embedder = RandomEmbedder(dim=16, seed=42)
    texts = [c.content for c in sample_chunks]
    embeddings = embedder.embed(texts)
    for chunk, emb in zip(sample_chunks, embeddings):
        chunk.embedding = emb
    return sample_chunks


@pytest.fixture()
def default_config() -> RAGConfig:
    return RAGConfig()


@pytest.fixture()
def tmp_text_file(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("Hello world. This is a test document.", encoding="utf-8")
    return str(f)
