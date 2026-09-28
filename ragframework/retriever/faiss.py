"""FAISS-backed approximate nearest-neighbour retrieval."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "FAISS support requires 'ragframework[faiss]'. "
    "Install it with: pip install ragframework[faiss]"
)


class FAISSRetriever(Retriever):
    """Retrieve chunks with an HNSW index and cosine similarity.

    Chunk and query vectors are L2-normalized before being passed to an
    ``IndexHNSWFlat`` configured for inner product. Inner product between
    normalized vectors is cosine similarity.

    Args:
        m: Number of HNSW neighbours stored per node.
        ef_construction: HNSW construction-time search depth.
        ef_search: HNSW query-time search depth.

    Raises:
        ImportError: If the optional ``faiss-cpu`` dependency is unavailable.
        RetrieverError: If an HNSW parameter is not positive.
    """

    def __init__(
        self,
        m: int = 32,
        ef_construction: int = 40,
        ef_search: int = 16,
    ) -> None:
        if not _FAISS_AVAILABLE:
            raise ImportError(_INSTALL_HINT)

        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in (m, ef_construction, ef_search)
        ):
            raise RetrieverError(
                "m, ef_construction, and ef_search must be positive integers."
            )

        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search

        self._chunks: list[Chunk] = []
        self._dimension: int | None = None
        self._index: Any | None = None

        # Maps logical chunk IDs to their current physical FAISS labels.
        # Old labels remain in the FAISS index as tombstones after replacement.
        self._id_to_index: dict[str, int] = {}

    def _build_index(
        self,
        vectors: np.ndarray[Any, Any],
        dimension: int,
    ) -> Any:
        """Build an HNSW index from normalized vectors."""
        index = faiss.IndexHNSWFlat(
            dimension,
            self._m,
            faiss.METRIC_INNER_PRODUCT,
        )

        index.hnsw.efConstruction = self._ef_construction
        index.hnsw.efSearch = self._ef_search

        index.add(vectors)

        return index

    def add(self, chunks: list[Chunk]) -> None:
        """Validate and upsert embedded chunks into the HNSW index.

        An empty batch is a no-op.

        Every vector in the incoming batch is validated before changing
        the Python chunk mapping.

        Chunks with an existing ID replace the previously indexed chunk.
        New IDs are appended.

        FAISS stores physical vectors permanently. When a chunk is replaced,
        its old physical vector becomes a tombstone and the ID mapping points
        to the newly appended vector.
        """
        if not chunks:
            return

        expected_dimension = self._dimension

        vectors: list[np.ndarray[Any, np.dtype[np.float32]]] = []

        # ---------------------------------------------------------------
        # 1. Validate the COMPLETE incoming batch first.
        # ---------------------------------------------------------------
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

            vectors.append(vector)

        assert expected_dimension is not None

        matrix = np.ascontiguousarray(
            np.stack(vectors),
            dtype=np.float32,
        )

        # ---------------------------------------------------------------
        # 2. Create the index for the first insertion.
        # ---------------------------------------------------------------
        if self._index is None:
            try:
                new_index = faiss.IndexHNSWFlat(
                    expected_dimension,
                    self._m,
                    faiss.METRIC_INNER_PRODUCT,
                )

                new_index.hnsw.efConstruction = self._ef_construction
                new_index.hnsw.efSearch = self._ef_search

                new_index.add(matrix)

            except Exception as exc:
                raise RetrieverError(
                    "FAISS failed to add chunk embeddings."
                ) from exc

            # Only update Python state after FAISS succeeds.
            self._index = new_index
            self._dimension = expected_dimension

            start_label = 0

            for offset, chunk in enumerate(chunks):
                self._id_to_index[chunk.id] = start_label + offset
                self._chunks.append(chunk)

            logger.debug(
                "Added chunks count=%d dimension=%d",
                len(chunks),
                matrix.shape[1],
            )

            return

        # ---------------------------------------------------------------
        # 3. Existing index.
        #
        # Append all incoming vectors to FAISS.
        #
        # This is intentionally done on the EXISTING index so a native
        # FAISS failure leaves the Python mapping unchanged.
        # ---------------------------------------------------------------
        start_label = int(self._index.ntotal)

        try:
            self._index.add(matrix)
        except Exception as exc:
            raise RetrieverError(
                "FAISS failed to add chunk embeddings."
            ) from exc

        # ---------------------------------------------------------------
        # 4. FAISS succeeded.
        #
        # Now update the logical mapping.
        # ---------------------------------------------------------------
        for offset, chunk in enumerate(chunks):
            physical_label = start_label + offset

            existing_logical_index = None

            for logical_index, existing_chunk in enumerate(self._chunks):
                if existing_chunk.id == chunk.id:
                    existing_logical_index = logical_index
                    break

            if existing_logical_index is not None:
                # Replace the logical chunk while preserving its position.
                self._chunks[existing_logical_index] = chunk
            else:
                # New logical chunk.
                self._chunks.append(chunk)

            # Point the ID to the newly appended physical vector.
            self._id_to_index[chunk.id] = physical_label

        self._dimension = expected_dimension

        logger.debug(
            "Added chunks count=%d dimension=%d",
            len(chunks),
            matrix.shape[1],
        )

        return

        # ---------------------------------------------------------------
        # 4. Upsert path.
        #
        # HNSW does not provide a reliable in-place "replace vector at
        # arbitrary logical position" operation for maintaining our
        # Python chunk mapping.
        #
        # Therefore we construct the desired final chunk list and rebuild
        # the index. Before rebuilding, we exercise the current FAISS
        # index's add operation so FAISS-native failures are surfaced
        # without changing the Python chunk mapping.
        # ---------------------------------------------------------------

        final_chunks = list(self._chunks)

        existing_indices = {
                chunk.id: index
                    for index, chunk in enumerate(self._chunks)
                }

        # Replace existing chunks in their original positions.
        for chunk in chunks:
            existing_index = existing_indices.get(chunk.id)

            if existing_index is not None:
                final_chunks[existing_index] = chunk

        # Append genuinely new chunks.
        for chunk in chunks:
            if chunk.id not in existing_indices:
                final_chunks.append(chunk)

        # ---------------------------------------------------------------
        # Build the final normalized matrix.
        # ---------------------------------------------------------------
        final_vectors: list[
            np.ndarray[Any, np.dtype[np.float32]]
        ] = []

        incoming_by_id: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {
            chunk.id: vector
            for chunk, vector in zip(chunks, vectors, strict=True)
        }

        for chunk in final_chunks:
            if chunk.id in incoming_by_id:
                final_vectors.append(incoming_by_id[chunk.id])
            else:
                if chunk.embedding is None:
                    raise RetrieverError(
                        f"Chunk '{chunk.id}' has no embedding. "
                        "Embed chunks before adding them to the retriever."
                    )

                final_vectors.append(
                    validate_vector(
                        chunk.embedding,
                        f"Chunk '{chunk.id}' embedding",
                    )
                )

        final_matrix = np.ascontiguousarray(
            np.stack(final_vectors),
            dtype=np.float32,
        )

        # ---------------------------------------------------------------
        # 5. Build replacement index before changing retriever state.
        # ---------------------------------------------------------------
        try:
            new_index = self._build_index(
                final_matrix,
                expected_dimension,
            )
        except Exception as exc:
            raise RetrieverError(
                "FAISS failed to add chunk embeddings."
            ) from exc

        # ---------------------------------------------------------------
        # 6. Commit only after FAISS successfully built the new index.
        # ---------------------------------------------------------------
        self._index = new_index
        self._dimension = expected_dimension
        self._chunks = final_chunks

        logger.debug(
            "Added chunks count=%d dimension=%d",
            len(chunks),
            final_matrix.shape[1],
        )

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[Chunk]:
        """Return active chunks ordered by approximate cosine similarity.

        Superseded physical FAISS vectors are ignored.

        When multiple active chunks have identical similarity scores,
        their logical insertion order is used as a deterministic
        tie-breaker.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise RetrieverError("top_k must be an integer.")

        if (
            top_k <= 0
            or self._index is None
            or self._dimension is None
            or not self._chunks
        ):
            return []

        query = validate_vector(
            query_embedding,
            "Query embedding",
        )

        if query.shape[0] != self._dimension:
            raise RetrieverError(
                f"Query embedding has dimension {query.shape[0]}; "
                f"expected {self._dimension}."
            )

        # Search all physical vectors because some of them may be
        # tombstones belonging to superseded chunks.
        count = int(self._index.ntotal)

        try:
            scores, labels = self._index.search(
                query.reshape(1, -1),
                count,
            )
        except Exception as exc:
            raise RetrieverError(
                "FAISS failed to search the index."
            ) from exc

        # Reverse mapping:
        # physical FAISS label -> logical chunk position.
        active_labels: dict[int, int] = {
            physical_label: logical_index
            for logical_index, chunk in enumerate(self._chunks)
            for physical_label in [self._id_to_index.get(chunk.id)]
            if physical_label is not None
        }

        candidates: list[tuple[float, int, Chunk]] = []

        for score, label in zip(
            scores[0],
            labels[0],
            strict=True,
        ):
            physical_label = int(label)

            if physical_label < 0:
                continue

            logical_index = active_labels.get(physical_label)

            # Ignore tombstoned / unknown physical labels.
            if logical_index is None:
                continue

            candidates.append(
                (
                    float(score),
                    logical_index,
                    self._chunks[logical_index],
                )
            )

        # Highest cosine similarity first.
        # Logical index breaks exact score ties deterministically.
        candidates.sort(
            key=lambda item: (-item[0], item[1])
        )

        return [
            chunk
            for _score, _logical_index, chunk in candidates[:top_k]
        ]

    def __len__(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)
