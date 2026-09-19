"""End-to-end tests for RAGPipeline."""

from __future__ import annotations

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

    def test_ingest_many_returns_total_chunk_count(self, pipeline, tmp_path):
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        first.write_text("alpha " * 30, encoding="utf-8")
        second.write_text("beta " * 30, encoding="utf-8")

        expected_count = sum(
            len(pipeline.chunker.chunk(pipeline.loader.load(str(source))[0]))
            for source in (first, second)
        )

        assert pipeline.ingest_many([str(first), str(second)]) == expected_count
        assert len(pipeline.retriever._chunks) == expected_count

    def test_ingest_many_reports_failed_source_and_success_count(self, pipeline, tmp_path):
        valid_source = tmp_path / "valid.txt"
        valid_source.write_text("valid content", encoding="utf-8")
        missing_source = tmp_path / "missing.txt"

        with pytest.raises(PipelineError) as exc_info:
            pipeline.ingest_many([str(valid_source), str(missing_source)])

        message = str(exc_info.value)
        assert str(missing_source) in message
        assert "1 source" in message
        assert "1 chunk" in message

    def test_ingest_missing_file_raises_pipeline_error(self, pipeline):
        with pytest.raises(PipelineError):
            pipeline.ingest("/no/such/file.txt")

    def test_query_accepts_top_k_override(self, pipeline, tmp_text_file):
        pipeline.ingest(tmp_text_file)

        response = pipeline.query("test", top_k=1)

        assert len(response.source_chunks) == 1

    def test_query_rejects_non_positive_top_k_override(self, pipeline):
        with pytest.raises(PipelineError, match="top_k must be positive"):
            pipeline.query("test", top_k=0)

    def test_query_response_includes_original_query(self, pipeline, tmp_text_file):
        query = "What is this about?"
        pipeline.ingest(tmp_text_file)

        response = pipeline.query(query)

        assert response.query == query

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
