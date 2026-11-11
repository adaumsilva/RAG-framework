"""In-memory retriever using cosine similarity."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ragframework.base import Chunk, Retriever
from ragframework.exceptions import RetrieverError
from ragframework.retriever._persistence import (
    check_format_version,
    parse_chunk_records,
    require_json_object,
)
from ragframework.utils.vectors import validate_vector

logger = logging.getLogger(__name__)
_FORMAT_VERSION = 1


def _validate_matrix(matrix: np.ndarray[Any, Any], directory: Path) -> np.ndarray[Any, Any]:
    """Return *matrix* as contiguous float32 if it is a 2-D array of finite numbers."""
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise RetrieverError(
            f"Malformed matrix.npy in '{directory}': expected a non-empty 2-D array, "
            f"got shape {matrix.shape}."
        )
    if not (np.issubdtype(matrix.dtype, np.floating) or np.issubdtype(matrix.dtype, np.integer)):
        raise RetrieverError(
            f"Malformed matrix.npy in '{directory}': expected numeric data, "
            f"got dtype {matrix.dtype}."
        )
    if not np.isfinite(matrix).all():
        raise RetrieverError(f"Malformed matrix.npy in '{directory}': values must all be finite.")
    return np.ascontiguousarray(matrix, dtype=np.float32)


class InMemoryRetriever(Retriever):
    """A simple retriever that stores chunks in RAM and ranks by cosine similarity.

    Suitable for small corpora and quick experiments. For large-scale use,
    swap this out with a FAISS or ChromaDB retriever (see
    ``.github/GOOD_FIRST_ISSUES.md``).
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray[Any, Any] | None = None  # shape (N, dim)
        self._matrix_blocks: list[np.ndarray[Any, Any]] = []
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
        self._matrix_blocks.append(matrix)

        self._matrix = None
        self._dimension = expected_dimension
        self._chunks.extend(chunks)
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
        if top_k <= 0 or not self._chunks:
            return []
        q = validate_vector(query_embedding, "Query embedding")
        if q.shape[0] != self._dimension:
            raise RetrieverError(
                f"Query embedding has dimension {q.shape[0]}; expected {self._dimension}."
            )
        if self._matrix is None:
            if len(self._matrix_blocks) == 1:
                # Reuse the existing array; do not create another copy.
                self._matrix = self._matrix_blocks[0]
            else:
                # Consolidate all blocks into one cached matrix and replace the
                # old block references so the original arrays can be released.
                self._matrix = np.concatenate(self._matrix_blocks)
                self._matrix_blocks = [self._matrix]

        scores: np.ndarray[Any, Any] = self._matrix @ q
        k = min(top_k, len(self._chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self._chunks[i] for i in top_indices]

    def clear(self) -> None:
        """Remove all chunks and reset the accepted embedding dimension."""
        self._chunks = []
        self._matrix = None
        self._matrix_blocks = []
        self._dimension = None

    def save(self, path: str | Path) -> None:
        """Write the index to ``path`` as ``matrix.npy`` and ``chunks.json``.

        Chunk ids, content, and metadata are stored as JSON; embeddings are
        stored only in ``matrix.npy``. The directory is created if needed and
        existing files are overwritten.

        Args:
            path: Directory to write the index files into.

        Raises:
            RetrieverError: If the retriever holds no chunks.
        """
        directory = Path(path)
        if len(self._chunks) == 0:
            raise RetrieverError("Cannot save an empty InMemoryRetriever")

        directory.mkdir(parents=True, exist_ok=True)

        matrix = np.concatenate(self._matrix_blocks)
        np.save(directory / "matrix.npy", matrix)

        records = [{"id": c.id, "content": c.content, "metadata": c.metadata} for c in self._chunks]
        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump({"format_version": _FORMAT_VERSION, "chunks": records}, f)

    @classmethod
    def load(cls, path: str | Path) -> InMemoryRetriever:
        """Rebuild a retriever previously written by :meth:`save`.

        Loaded chunks have ``embedding=None``; their vectors are restored from
        ``matrix.npy``. Pickled data is never loaded.

        Args:
            path: Directory that :meth:`save` wrote to.

        Returns:
            A new retriever containing the saved chunks.

        Raises:
            RetrieverError: If the directory or a file is missing or corrupt,
                ``chunks.json`` does not have the expected structure, the
                format version does not match, ``matrix.npy`` is not a
                two-dimensional array of finite numbers, or the number of
                chunks does not match the number of stored vectors.
        """
        directory = Path(path)
        try:
            raw_matrix = np.load(directory / "matrix.npy", allow_pickle=False)
            with open(directory / "chunks.json", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError) as exc:
            raise RetrieverError(
                f"Could not load InMemoryRetriever from '{directory}': {exc}"
            ) from exc

        data = require_json_object(data, "chunks.json", directory)
        check_format_version(data.get("format_version"), _FORMAT_VERSION, "chunks.json", directory)
        chunks = parse_chunk_records(data.get("chunks"), directory)
        matrix = _validate_matrix(raw_matrix, directory)
        if len(chunks) != matrix.shape[0]:
            raise RetrieverError(
                f"'{directory}' is inconsistent: {len(chunks)} chunks "
                f"but {matrix.shape[0]} vectors."
            )

        retriever = cls()
        retriever._chunks = chunks
        retriever._matrix_blocks = [matrix]
        retriever._dimension = int(matrix.shape[1])
        return retriever

    def __len__(self) -> int:
        return len(self._chunks)
