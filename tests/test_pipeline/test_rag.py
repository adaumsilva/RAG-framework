"""End-to-end tests for RAGPipeline."""

from __future__ import annotations

import pytest

from ragframework.base import Chunk, Embedder, Generator, Reranker
from ragframework.config import RAGConfig
from ragframework.document.chunkers import FixedSizeChunker
from ragframework.document.loaders import TextFileLoader
from ragframework.embeddings.random_embedder import RandomEmbedder
from ragframework.exceptions import PipelineError
from ragframework.generator.echo_generator import EchoGenerator
from ragframework.pipeline.rag import RAGPipeline
from ragframework.retriever.in_memory import InMemoryRetriever


class ReverseReranker(Reranker):
    def rerank(self, query, chunks, top_k):
        return list(reversed(chunks))[:top_k]


class RecordingGenerator(Generator):
    def __init__(self):
        self.context = []

    def generate(self, query, context):
        self.context = list(context)
        return "ok"


class MixedDimensionEmbedder(Embedder):
    def embed(self, texts):
        embeddings = [[0.0] * 8 for _ in texts]
        if len(embeddings) > 1:
            embeddings[1] = [0.0] * 3
        return embeddings


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


class ShortSecondCallEmbedder(Embedder):
    """Embedder that returns too few vectors on the second call."""

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        embeddings = [[float(index + 1)] * 16 for index, _ in enumerate(texts)]
        if self.calls == 2:
            return embeddings[:-1]
        return embeddings


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

    def test_ingest_many_wraps_unexpected_ingest_error_with_progress(self, tmp_path):
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        first.write_text("alpha " * 30, encoding="utf-8")
        second.write_text("beta " * 30, encoding="utf-8")
        p = RAGPipeline(
            loader=TextFileLoader(),
            chunker=FixedSizeChunker(chunk_size=50, chunk_overlap=10),
            embedder=ShortSecondCallEmbedder(),
            retriever=InMemoryRetriever(),
            generator=EchoGenerator(),
        )
        first_count = len(p.chunker.chunk(p.loader.load(str(first))[0]))

        with pytest.raises(PipelineError) as exc_info:
            p.ingest_many([str(first), str(second)])

        assert isinstance(exc_info.value.__cause__, ValueError)
        message = str(exc_info.value)
        assert str(second) in message
        assert "1 source" in message
        assert f"{first_count} chunks" in message
        assert len(p.retriever._chunks) == first_count

    def test_query_with_reranker_uses_top_k_override_for_final_limit(self):
        class FixedRetriever(InMemoryRetriever):
            def retrieve(self, query_embedding, top_k=5):
                return [Chunk(id=str(i), content=f"chunk {i}") for i in range(top_k)]

        generator = RecordingGenerator()
        p = RAGPipeline(
            loader=TextFileLoader(),
            chunker=FixedSizeChunker(),
            embedder=RandomEmbedder(dim=16, seed=0),
            retriever=FixedRetriever(),
            generator=generator,
            reranker=ReverseReranker(),
            config=RAGConfig(top_k=3),
        )

        response = p.query("test", top_k=1)

        assert [chunk.id for chunk in generator.context] == ["3"]
        assert [chunk.id for chunk in response.source_chunks] == ["3"]

    def test_query_without_reranker_uses_top_k_override_for_final_limit(self):
        class FixedRetriever(InMemoryRetriever):
            def retrieve(self, query_embedding, top_k=5):
                return [Chunk(id=str(i), content=f"chunk {i}") for i in range(top_k)]

        generator = RecordingGenerator()
        p = RAGPipeline(
            loader=TextFileLoader(),
            chunker=FixedSizeChunker(),
            embedder=RandomEmbedder(dim=16, seed=0),
            retriever=FixedRetriever(),
            generator=generator,
            config=RAGConfig(top_k=3, retrieve_k=8),
        )

        response = p.query("test", top_k=1)

        assert [chunk.id for chunk in generator.context] == ["0"]
        assert [chunk.id for chunk in response.source_chunks] == ["0"]

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

    def test_ingest_validates_every_embedding_before_indexing(self, tmp_text_file):
        retriever = InMemoryRetriever()
        p = RAGPipeline(
            loader=TextFileLoader(),
            chunker=FixedSizeChunker(chunk_size=20, chunk_overlap=0),
            embedder=MixedDimensionEmbedder(),
            retriever=retriever,
            generator=EchoGenerator(),
            config=RAGConfig(embedding_dim=8),
        )

        with pytest.raises(
            PipelineError,
            match=r"Embedder produced 3-dim vectors but RAGConfig\.embedding_dim is 8",
        ):
            p.ingest(tmp_text_file)

        assert len(retriever) == 0

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

    def test_from_config_preserves_reranker(self):
        config = RAGConfig(embedding_dim=16)
        reranker = ReverseReranker()

        p = RAGPipeline.from_config(
            config,
            loader=TextFileLoader(),
            embedder=RandomEmbedder(dim=16),
            retriever=InMemoryRetriever(),
            generator=EchoGenerator(),
            reranker=reranker,
        )

        assert p.reranker is reranker

    def test_ingest_embeds_in_configured_batches(self, tmp_path):
        source = tmp_path / "many.txt"
        source.write_text("word " * 80)

        class RecordingEmbedder(Embedder):
            def __init__(self):
                self.call_sizes: list[int] = []

            def embed(self, texts):
                self.call_sizes.append(len(texts))
                return [[1.0] + [0.0] * 7 for _ in texts]

        embedder = RecordingEmbedder()
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
        p = RAGPipeline(
            loader=TextFileLoader(),
            chunker=chunker,
            embedder=embedder,
            retriever=InMemoryRetriever(),
            generator=EchoGenerator(),
            config=RAGConfig(embed_batch_size=5),
        )

        count = p.ingest(str(source))
        assert count > 0
        assert all(size <= 5 for size in embedder.call_sizes)
        assert sum(embedder.call_sizes) == count
        assert embedder.call_sizes == [5] * (count // 5) + ([count % 5] if count % 5 else [])
