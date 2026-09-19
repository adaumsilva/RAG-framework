"""No-op reranker."""

from ragframework.base import Chunk, Reranker


class NoOpReranker(Reranker):
    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        return chunks[:top_k]
