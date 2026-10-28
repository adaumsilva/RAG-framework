"""In-memory retriever using cosine similarity."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

from ragframework.base import Chunk, Retriever
from ragframework.exceptions import RetrieverError
from ragframework.utils.vectors import validate_vector

logger = logging.getLogger(__name__)


class InMemoryRetriever(Retriever):
    """Store chunks in RAM and rank them by cosine similarity."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._id_to_index: dict[str, int] = {}
        self._matrix: np.ndarray[Any, Any] | None = None
        self._matrix_blocks: list[np.ndarray[Any, Any]] = []
        self._dimension: int | None = None

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks using upsert semantics.

        Existing chunk IDs are replaced. If an ID occurs multiple times
        in a batch, the last occurrence wins. The entire batch is
        validated before stored chunks are modified.
        """
        if not chunks:
            return

        expected_dimension = self._dimension
        validated: list[tuple[Chunk, np.ndarray[Any, Any]]] = []

        # Validate the entire batch before changing stored data.
        for chunk in chunks:
            if chunk.embedding is None:
                raise RetrieverError(
                    f"Chunk '{chunk.id}' has no embedding. "
                    "Embed chunks before adding them to the retriever."
                )

            vector = validate_vector(
                chunk.embedding,
                f"Chunk '{chunk.id}' embedding",
            )

            if expected_dimension is None:
                expected_dimension = int(vector.shape[0])
            elif vector.shape[0] != expected_dimension:
                raise RetrieverError(
                    f"Chunk '{chunk.id}' embedding has dimension "
                    f"{vector.shape[0]}; expected {expected_dimension}."
                )

            validated.append((chunk, vector))

        # Deduplicate within the batch; the last occurrence wins.
        latest: dict[str, tuple[Chunk, np.ndarray[Any, Any]]] = {}
        for chunk, vector in validated:
            latest[chunk.id] = (chunk, vector)

        has_existing_id = any(chunk_id in self._id_to_index for chunk_id in latest)

        # Keep a consolidated matrix as a block when invalidating its cache.
        if self._matrix is not None:
            self._matrix_blocks = [self._matrix]
            self._matrix = None

        # Replacements need a single writable matrix containing all old rows.
        if has_existing_id and self._matrix_blocks:
            self._matrix = np.concatenate(self._matrix_blocks, axis=0)
            self._matrix_blocks = []

        new_vectors: list[np.ndarray[Any, Any]] = []

        for chunk_id, (chunk, vector) in latest.items():
            existing_index = self._id_to_index.get(chunk_id)

            if existing_index is None:
                self._id_to_index[chunk_id] = len(self._chunks)
                self._chunks.append(chunk)
                new_vectors.append(vector)
            else:
                self._chunks[existing_index] = chunk

                # A replacement must update the existing row, not append one.
                if self._matrix is None:
                    if self._matrix_blocks:
                        self._matrix = np.concatenate(self._matrix_blocks, axis=0)
                    else:
                        self._matrix = np.empty(
                            (0, expected_dimension or 0),
                            dtype=np.float32,
                        )
                    self._matrix_blocks = []

                self._matrix[existing_index] = vector

        # Store all new vectors from this add() call in one block.
        if new_vectors:
            new_block = np.stack(new_vectors, axis=0)

            if self._matrix is not None:
                self._matrix_blocks = [self._matrix, new_block]
                self._matrix = None
            else:
                self._matrix_blocks.append(new_block)

        # Invalidate the cache after replacements so retrieval rebuilds it.
        elif self._matrix is not None:
            self._matrix_blocks = [self._matrix]
            self._matrix = None

        self._dimension = expected_dimension

        logger.debug(
            "Added chunks count=%d dimension=%d",
            len(chunks),
            expected_dimension,
        )

    @staticmethod
    def _validate_filter(filter: Mapping[str, Any] | None) -> None:
        """Validate that filter values are supported scalar types."""
        if filter is None:
            return

        if not isinstance(filter, Mapping):
            raise RetrieverError("filter must be a mapping or None.")

        for key, value in filter.items():
            if not isinstance(key, str):
                raise RetrieverError("Filter keys must be strings.")

            if not isinstance(value, (str, int, float, bool)):
                raise RetrieverError(
                    f"Filter value for '{key}' must be a scalar (str, int, float, or bool)."
                )

            if isinstance(value, float) and not np.isfinite(value):
                raise RetrieverError(f"Filter value for '{key}' must be finite.")

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        *,
        filter: Mapping[str, Any] | None = None,
    ) -> list[Chunk]:
        """Rank chunks by cosine similarity, optionally filtering metadata."""
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise RetrieverError("top_k must be an integer.")

        self._validate_filter(filter)

        if top_k <= 0 or not self._chunks:
            return []

        query = validate_vector(query_embedding, "Query embedding")

        if query.shape[0] != self._dimension:
            raise RetrieverError(
                f"Query embedding has dimension {query.shape[0]}; expected {self._dimension}."
            )

        if self._matrix is None:
            if not self._matrix_blocks:
                self._matrix = np.empty(
                    (0, self._dimension or 0),
                    dtype=np.float32,
                )
            elif len(self._matrix_blocks) == 1:
                self._matrix = self._matrix_blocks[0]
            else:
                self._matrix = np.concatenate(self._matrix_blocks, axis=0)

            # Keep one canonical block after consolidation.
            self._matrix_blocks = [self._matrix]

        scores: np.ndarray[Any, Any] = self._matrix @ query
        ordered_indices = np.argsort(scores)[::-1]

        results: list[Chunk] = []

        for index in ordered_indices:
            chunk = self._chunks[int(index)]

            if filter is not None and not all(
                chunk.metadata.get(key) == expected for key, expected in filter.items()
            ):
                continue

            results.append(chunk)

            if len(results) == top_k:
                break

        return results

    def clear(self) -> None:
        """Remove all chunks and reset the accepted embedding dimension."""
        self._chunks = []
        self._id_to_index = {}
        self._matrix = None
        self._matrix_blocks = []
        self._dimension = None

    def __len__(self) -> int:
        return len(self._chunks)
