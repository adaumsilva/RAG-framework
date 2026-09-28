"""In-memory retriever using cosine similarity."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ragframework.base import Chunk, Retriever
from ragframework.exceptions import RetrieverError
from ragframework.utils.vectors import validate_vector

logger = logging.getLogger(__name__)


class InMemoryRetriever(Retriever):
    """A simple retriever that stores chunks in RAM and ranks by cosine similarity.

    Suitable for small corpora and quick experiments. For large-scale use,
    swap this out with a FAISS or ChromaDB retriever (see
    ``.github/GOOD_FIRST_ISSUES.md``).
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray[Any, Any] | None = None  # shape (N, dim)
        self._id_to_index: dict[str, int] = {}
        self._dimension: int | None = None

    def add(self, chunks: list[Chunk]) -> None:
        """Validate and upsert a batch before changing the stored chunks.

        An empty batch is a no-op. Invalid vectors or inconsistent dimensions
        raise ``RetrieverError`` without adding or replacing any part of the
        batch.
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

        # Build the initial index.
        if self._matrix is None:
            self._matrix = matrix
            self._dimension = expected_dimension

            for chunk in chunks:
                self._id_to_index[chunk.id] = len(self._chunks)
                self._chunks.append(chunk)

            logger.debug(
                "Added chunks count=%d dimension=%d",
                len(chunks),
                matrix.shape[1],
            )
            return

        # Apply replacements and collect genuinely new chunks.
        new_chunks: list[Chunk] = []
        new_vectors: list[np.ndarray[Any, np.dtype[np.float32]]] = []

        for chunk, vector in zip(chunks, vectors, strict=True):
            existing_index = self._id_to_index.get(chunk.id)

            if existing_index is not None:
                self._chunks[existing_index] = chunk
                self._matrix[existing_index] = vector
            else:
                new_chunks.append(chunk)
                new_vectors.append(vector)

        if new_chunks:
            start_index = len(self._chunks)
            self._chunks.extend(new_chunks)

            self._matrix = np.concatenate(
                (self._matrix, np.stack(new_vectors))
            )

            for offset, chunk in enumerate(new_chunks):
                self._id_to_index[chunk.id] = start_index + offset

        self._dimension = expected_dimension

        logger.debug(
            "Added chunks count=%d dimension=%d",
            len(chunks),
            matrix.shape[1],
        )

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Rank chunks by cosine similarity.

        Non-positive ``top_k`` values return an empty list.
        An empty index also returns an empty list. Invalid query vectors
        or dimensions raise ``RetrieverError``. Non-integer / bool
        ``top_k`` raises ``RetrieverError``.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise RetrieverError("top_k must be an integer.")
        if top_k <= 0 or not self._chunks or self._matrix is None:
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
