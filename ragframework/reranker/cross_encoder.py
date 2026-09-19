"""Sentence Transformers cross-encoder reranker."""

from __future__ import annotations

from ragframework.base import Chunk, Reranker
from ragframework.exceptions import RerankerError


class CrossEncoderReranker(Reranker):
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "Cross-encoder reranking requires 'ragframework[huggingface]'"
            ) from exc
        try:
            self._model = CrossEncoder(model_name, device=device)
        except Exception as exc:
            raise RerankerError(f"Could not load reranker model: {exc}") from exc

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        try:
            scores = self._model.predict([(query, chunk.content) for chunk in chunks])
        except Exception as exc:
            raise RerankerError(f"Could not rerank chunks: {exc}") from exc
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda item: item[1], reverse=True)
        for chunk, score in ranked[:top_k]:
            chunk.metadata["rerank_score"] = float(score)
        return [chunk for chunk, _ in ranked[:top_k]]
