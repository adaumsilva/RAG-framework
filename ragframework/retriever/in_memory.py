"""In-memory retriever using cosine similarity."""

from __future__ import annotations

from typing import Any

import numpy as np

from ragframework.base import Chunk, Retriever
from ragframework.exceptions import RetrieverError


class InMemoryRetriever(Retriever):
    """A simple retriever that stores chunks in RAM and ranks by cosine similarity.

    Suitable for small corpora and quick experiments. For large-scale use,
    swap this out with a FAISS or ChromaDB retriever (see
    ``.github/GOOD_FIRST_ISSUES.md``).
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray[Any, Any] | None = None  # shape (N, dim)

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            if chunk.embedding is None:
                raise RetrieverError(
                    f"Chunk '{chunk.id}' has no embedding. "
                    "Embed chunks before adding them to the retriever."
                )
        self._chunks.extend(chunks)
        vectors = np.array([c.embedding for c in self._chunks], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._matrix = vectors / norms

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        if not self._chunks or self._matrix is None:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(q) or 1.0
        q = q / norm
        scores: np.ndarray[Any, Any] = self._matrix @ q
        k = min(top_k, len(self._chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self._chunks[i] for i in top_indices]

    def __len__(self) -> int:
        return len(self._chunks)
