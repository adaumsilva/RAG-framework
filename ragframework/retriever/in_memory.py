"""In-memory retriever using cosine similarity."""

from __future__ import annotations

from typing import Any

import numpy as np

from ragframework.base import Chunk, Retriever
from ragframework.exceptions import RetrieverError
from ragframework.utils.vectors import validate_vector


class InMemoryRetriever(Retriever):
    """A simple retriever that stores chunks in RAM and ranks by cosine similarity.

    Suitable for small corpora and quick experiments. For large-scale use,
    swap this out with a FAISS or ChromaDB retriever (see
    ``.github/GOOD_FIRST_ISSUES.md``).
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray[Any, Any] | None = None  # shape (N, dim)
        self._dimension: int | None = None

    def add(self, chunks: list[Chunk]) -> None:
        """Validate and normalize a batch before changing the stored chunks.

        An empty batch is a no-op. Invalid vectors or inconsistent dimensions
        raise ``RetrieverError`` without adding any part of the batch.
        """
        if not chunks:
            return

        expected_dimension = self._dimension
        vectors: list[np.ndarray[Any, np.dtype[np.float32]]] = []
        for chunk in chunks:
            if chunk.embedding is None:
                raise RetrieverError(
                    f"Chunk '{chunk.id}' has no embedding. "
                    "Embed chunks before adding them to the retriever."
                )
            vector = validate_vector(chunk.embedding, f"Chunk '{chunk.id}' embedding")
            if expected_dimension is None:
                expected_dimension = int(vector.shape[0])
            elif vector.shape[0] != expected_dimension:
                raise RetrieverError(
                    f"Chunk '{chunk.id}' embedding has dimension {vector.shape[0]}; "
                    f"expected {expected_dimension}."
                )
            vectors.append(vector)

        matrix = np.stack(vectors)
        if self._matrix is not None:
            matrix = np.concatenate((self._matrix, matrix))

        self._matrix = matrix
        self._dimension = expected_dimension
        self._chunks.extend(chunks)

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Rank chunks by cosine similarity, returning an empty list for an empty index.

        Invalid query vectors or dimensions raise ``RetrieverError``.
        """
        if not self._chunks or self._matrix is None:
            return []
        q = validate_vector(query_embedding, "Query embedding")
        if q.shape[0] != self._dimension:
            raise RetrieverError(
                f"Query embedding has dimension {q.shape[0]}; expected {self._dimension}."
            )
        scores: np.ndarray[Any, Any] = self._matrix @ q
        k = min(top_k, len(self._chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self._chunks[i] for i in top_indices]

    def __len__(self) -> int:
        return len(self._chunks)
