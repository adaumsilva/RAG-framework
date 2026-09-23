"""FAISS-backed approximate nearest-neighbour retrieval."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import faiss
except ImportError:
    _FAISS_AVAILABLE = False
else:
    _FAISS_AVAILABLE = True

from ragframework.base import Chunk, Retriever
from ragframework.exceptions import RetrieverError
from ragframework.utils.vectors import validate_vector

_INSTALL_HINT = (
    "FAISS support requires 'ragframework[faiss]'. Install it with: pip install ragframework[faiss]"
)


class FAISSRetriever(Retriever):
    """Retrieve chunks with an HNSW index and cosine similarity.

    Chunk and query vectors are L2-normalized before being passed to an
    ``IndexHNSWFlat`` configured for inner product.  Inner product between
    normalized vectors is cosine similarity.

    Args:
        m: Number of HNSW neighbours stored per node.
        ef_construction: HNSW construction-time search depth.
        ef_search: HNSW query-time search depth.

    Raises:
        ImportError: If the optional ``faiss-cpu`` dependency is unavailable.
        RetrieverError: If an HNSW parameter is not positive.
    """

    def __init__(self, m: int = 32, ef_construction: int = 40, ef_search: int = 16) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError(_INSTALL_HINT)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (m, ef_construction, ef_search)
        ):
            raise RetrieverError("m, ef_construction, and ef_search must be positive integers.")

        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._chunks: list[Chunk] = []
        self._dimension: int | None = None
        self._index: Any | None = None

    def add(self, chunks: list[Chunk]) -> None:
        """Validate, normalize, and add embedded chunks to the HNSW index.

        An empty batch is a no-op.  Every vector in a batch is validated before
        either the FAISS index or the chunk mapping is changed.
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

        assert expected_dimension is not None
        matrix = np.ascontiguousarray(np.stack(vectors), dtype=np.float32)
        new_index = self._index
        try:
            if new_index is None:
                new_index = faiss.IndexHNSWFlat(
                    expected_dimension, self._m, faiss.METRIC_INNER_PRODUCT
                )
                new_index.hnsw.efConstruction = self._ef_construction
                new_index.hnsw.efSearch = self._ef_search
            new_index.add(matrix)
        except Exception as exc:
            raise RetrieverError("FAISS failed to add chunk embeddings.") from exc

        self._index = new_index
        self._dimension = expected_dimension
        self._chunks.extend(chunks)

    def retrieve(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Return up to ``top_k`` chunks ordered by approximate cosine similarity.

        Non-positive ``top_k`` values return an empty list.  An empty index also
        returns an empty list.  Invalid FAISS labels (for example ``-1`` when
        fewer than ``top_k`` neighbours exist) are ignored.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise RetrieverError("top_k must be an integer.")
        if top_k <= 0 or self._index is None or self._dimension is None:
            return []

        query = validate_vector(query_embedding, "Query embedding")
        if query.shape[0] != self._dimension:
            raise RetrieverError(
                f"Query embedding has dimension {query.shape[0]}; expected {self._dimension}."
            )

        count = min(top_k, len(self._chunks))
        try:
            _scores, labels = self._index.search(query.reshape(1, -1), count)
        except Exception as exc:
            raise RetrieverError("FAISS failed to search the index.") from exc

        return [
            self._chunks[int(label)] for label in labels[0] if 0 <= int(label) < len(self._chunks)
        ]

    def __len__(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)
