"""Tests for AsyncRAGPipeline."""

import pytest

from ragframework.config import RAGConfig
from ragframework.document.chunkers import FixedSizeChunker
from ragframework.document.loaders import TextFileLoader
from ragframework.embeddings.random_embedder import RandomEmbedder
from ragframework.generator.echo_generator import EchoGenerator
from ragframework.pipeline.async_rag import AsyncRAGPipeline
from ragframework.pipeline.rag import RAGPipeline
from ragframework.retriever.in_memory import InMemoryRetriever


@pytest.fixture()
def pipeline(tmp_text_file):
    return RAGPipeline(
        loader=TextFileLoader(),
        chunker=FixedSizeChunker(chunk_size=50, chunk_overlap=10),
        embedder=RandomEmbedder(dim=16, seed=0),
        retriever=InMemoryRetriever(),
        generator=EchoGenerator(),
        config=RAGConfig(top_k=2),
    )


@pytest.mark.asyncio
async def test_async_ingest(pipeline, tmp_text_file):
    async_pipeline = AsyncRAGPipeline.from_pipeline(pipeline)

    count = await async_pipeline.async_ingest(tmp_text_file)

    assert count > 0


@pytest.mark.asyncio
async def test_async_query(pipeline, tmp_text_file):
    async_pipeline = AsyncRAGPipeline.from_pipeline(pipeline)

    await async_pipeline.async_ingest(tmp_text_file)
    response = await async_pipeline.async_query("What is this about?")

    assert response.answer
    assert isinstance(response.source_chunks, list)


def test_config_is_forwarded(pipeline):
    async_pipeline = AsyncRAGPipeline.from_pipeline(pipeline)

    assert async_pipeline.config.top_k == 2


def test_from_pipeline_wraps_existing_pipeline(pipeline):
    async_pipeline = AsyncRAGPipeline.from_pipeline(pipeline)

    assert async_pipeline._pipeline is pipeline
