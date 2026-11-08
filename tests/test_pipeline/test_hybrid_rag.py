"""Integration tests for RAGPipeline with hybrid retrieval."""

from ragframework.base import Chunk, Document, DocumentLoader, Embedder, Generator, TextChunker
from ragframework.config import RAGConfig
from ragframework.pipeline.rag import RAGPipeline
from ragframework.retriever.bm25 import BM25Retriever
from ragframework.retriever.hybrid import HybridRetriever
from ragframework.retriever.in_memory import InMemoryRetriever


class StaticLoader(DocumentLoader):
    def load(self, source: str) -> list[Document]:
        return [Document(id=source, content=source)]


class StaticChunker(TextChunker):
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks

    def chunk(self, document: Document) -> list[Chunk]:
        return self.chunks


class RoutingEmbedder(Embedder):
    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = {
            "SKU-42 red widget": [1.0, 0.0],
            "red widget with matching semantics": [0.9, 0.1],
            "SKU-42 catalog entry": [0.0, 1.0],
            "sku-42": [1.0, 0.0],
        }
        return [vectors[text] for text in texts]


class RecordingGenerator(Generator):
    def __init__(self) -> None:
        self.context: list[Chunk] = []

    def generate(self, query: str, context: list[Chunk]) -> str:
        self.context = context
        return "ok"


def test_pipeline_uses_hybrid_retrieval() -> None:
    chunks = [
        Chunk(id="both", content="SKU-42 red widget"),
        Chunk(id="dense-only", content="red widget with matching semantics"),
        Chunk(id="sparse-only", content="SKU-42 catalog entry"),
    ]
    generator = RecordingGenerator()
    retriever = HybridRetriever(
        dense=InMemoryRetriever(),
        sparse=BM25Retriever(),
    )
    pipeline = RAGPipeline(
        loader=StaticLoader(),
        chunker=StaticChunker(chunks),
        embedder=RoutingEmbedder(),
        retriever=retriever,
        generator=generator,
        config=RAGConfig(top_k=1),
    )

    assert pipeline.ingest("document") == 3
    response = pipeline.query("sku-42")

    assert [chunk.id for chunk in response.source_chunks] == ["both"]
    assert [chunk.id for chunk in generator.context] == ["both"]
