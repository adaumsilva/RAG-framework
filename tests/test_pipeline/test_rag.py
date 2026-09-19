"""End-to-end tests for RAGPipeline."""

import pytest

from ragframework.config import RAGConfig
from ragframework.document.chunkers import FixedSizeChunker
from ragframework.document.loaders import TextFileLoader
from ragframework.embeddings.random_embedder import RandomEmbedder
from ragframework.exceptions import PipelineError
from ragframework.generator.echo_generator import EchoGenerator
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


class TestRAGPipeline:
    def test_ingest_returns_chunk_count(self, pipeline, tmp_text_file):
        count = pipeline.ingest(tmp_text_file)
        assert count > 0

    def test_query_returns_response(self, pipeline, tmp_text_file):
        pipeline.ingest(tmp_text_file)
        response = pipeline.query("What is this about?")
        assert response.answer
        assert isinstance(response.source_chunks, list)

    def test_query_top_k_respected(self, pipeline, tmp_text_file):
        pipeline.ingest(tmp_text_file)
        response = pipeline.query("test")
        assert len(response.source_chunks) <= 2

    def test_ingest_missing_file_raises_pipeline_error(self, pipeline):
        with pytest.raises(PipelineError):
            pipeline.ingest("/no/such/file.txt")

    def test_query_empty_index_returns_no_context_message(self):
        p = RAGPipeline(
            loader=TextFileLoader(),
            chunker=FixedSizeChunker(),
            embedder=RandomEmbedder(dim=16, seed=0),
            retriever=InMemoryRetriever(),
            generator=EchoGenerator(),
        )
        response = p.query("anything")
        assert "No context" in response.answer

    def test_default_config_used_when_none_given(self):
        p = RAGPipeline(
            loader=TextFileLoader(),
            chunker=FixedSizeChunker(),
            embedder=RandomEmbedder(dim=16),
            retriever=InMemoryRetriever(),
            generator=EchoGenerator(),
        )
        assert p.config.top_k == 5

    def test_embedding_dimension_mismatch_during_ingest(self, tmp_text_file):
        p = RAGPipeline(
            loader=TextFileLoader(),
            chunker=FixedSizeChunker(),
            embedder=RandomEmbedder(dim=16),
            retriever=InMemoryRetriever(),
            generator=EchoGenerator(),
            config=RAGConfig(embedding_dim=8),
        )

        with pytest.raises(
            PipelineError,
            match=r"Embedder produced 16-dim vectors but RAGConfig\.embedding_dim is 8",
        ):
            p.ingest(tmp_text_file)

    def test_embedding_dimension_mismatch_during_query(self):
        p = RAGPipeline(
            loader=TextFileLoader(),
            chunker=FixedSizeChunker(),
            embedder=RandomEmbedder(dim=16),
            retriever=InMemoryRetriever(),
            generator=EchoGenerator(),
            config=RAGConfig(embedding_dim=8),
        )

        with pytest.raises(
            PipelineError,
            match=r"Embedder produced 16-dim vectors but RAGConfig\.embedding_dim is 8",
        ):
            p.query("test")

    def test_embedding_dimension_none_disables_validation(self, pipeline, tmp_text_file):
        assert pipeline.config.embedding_dim is None
        assert pipeline.ingest(tmp_text_file) > 0

    def test_from_config_builds_chunker(self):
        config = RAGConfig(chunk_size=37, chunk_overlap=9, embedding_dim=16)

        p = RAGPipeline.from_config(
            config,
            loader=TextFileLoader(),
            embedder=RandomEmbedder(dim=16),
            retriever=InMemoryRetriever(),
            generator=EchoGenerator(),
        )

        assert isinstance(p.chunker, FixedSizeChunker)
        assert p.chunker.chunk_size == 37
        assert p.chunker.chunk_overlap == 9
        assert p.config is config
