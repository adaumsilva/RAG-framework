"""FAISS-backed approximate nearest-neighbour retrieval."""

from __future__ import annotations

import json
import logging
from pathlib import Path
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
from ragframework.retriever._persistence import (
    check_format_version,
    parse_chunk_records,
    require_json_object,
    require_positive_int,
)
from ragframework.utils.vectors import validate_vector

logger = logging.getLogger(__name__)
_INSTALL_HINT = (
    "FAISS support requires 'ragframework[faiss]'. Install it with: pip install ragframework[faiss]"
)
_FORMAT_VERSION = 1


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
        logger.debug(
            "Added chunks count=%d dimension=%d",
            len(chunks),
            matrix.shape[1],
        )

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

    def clear(self) -> None:
        """Drop the index so the next batch can use a new dimension."""
        self._chunks = []
        self._dimension = None
        self._index = None

    def save(self, path: str | Path) -> None:
        """Write ``index.faiss``, ``chunks.json`` and ``meta.json`` to ``path``.

        ``meta.json`` records the format version, dimension, and HNSW
        parameters (``m``, ``ef_construction``, ``ef_search``). The directory
        is created if needed and existing files are overwritten.

        Args:
            path: Directory to write the index files into.

        Raises:
            RetrieverError: If the retriever holds no chunks.
        """
        if self._index is None:
            raise RetrieverError("Cannot save an empty FAISSRetriever.")
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(directory / "index.faiss"))

        records = [{"id": c.id, "content": c.content, "metadata": c.metadata} for c in self._chunks]
        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(records, f)

        meta = {
            "format_version": _FORMAT_VERSION,
            "dimension": self._dimension,
            "m": self._m,
            "ef_construction": self._ef_construction,
            "ef_search": self._ef_search,
        }
        with open(directory / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)

    @classmethod
    def load(cls, path: str | Path) -> FAISSRetriever:
        """Rebuild a retriever previously written by :meth:`save`.

        The HNSW parameters are restored from ``meta.json``. Loaded chunks
        have ``embedding=None``; their vectors live in ``index.faiss``.

        Warning:
            ``index.faiss`` is parsed by FAISS's native code. Only load
            indexes that you created or obtained from a trusted source and
            that have not been tampered with.

        Args:
            path: Directory that :meth:`save` wrote to.

        Returns:
            A new retriever containing the saved index and chunks.

        Raises:
            ImportError: If the optional ``faiss-cpu`` dependency is unavailable.
            RetrieverError: If the directory or a file is missing or corrupt,
                ``meta.json`` or ``chunks.json`` does not have the expected
                structure, the format version does not match, or the index
                dimension, vector count, or metric does not match the
                stored metadata and chunks.
        """
        if not _FAISS_AVAILABLE:
            raise ImportError(_INSTALL_HINT)

        directory = Path(path)
        try:
            with open(directory / "meta.json", encoding="utf-8") as f:
                raw_meta = json.load(f)
            with open(directory / "chunks.json", encoding="utf-8") as f:
                records = json.load(f)
        except (OSError, ValueError) as exc:
            raise RetrieverError(
                f"Could not load FAISSRetriever from '{directory}': {exc}"
            ) from exc

        # Validate the JSON files before handing index.faiss to the native parser.
        meta = require_json_object(raw_meta, "meta.json", directory)
        check_format_version(meta.get("format_version"), _FORMAT_VERSION, "meta.json", directory)
        dimension = require_positive_int(meta, "dimension", directory)
        m = require_positive_int(meta, "m", directory)
        ef_construction = require_positive_int(meta, "ef_construction", directory)
        ef_search = require_positive_int(meta, "ef_search", directory)
        chunks = parse_chunk_records(records, directory)

        try:
            index = faiss.read_index(str(directory / "index.faiss"))
        except (OSError, RuntimeError) as exc:
            raise RetrieverError(
                f"Could not load FAISSRetriever from '{directory}': {exc}"
            ) from exc

        if index.d != dimension:
            raise RetrieverError(
                f"'{directory}' is inconsistent: index.faiss has dimension {index.d} "
                f"but meta.json says {dimension}."
            )
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise RetrieverError(
                f"'{directory}' is inconsistent: index.faiss does not use the "
                "inner-product metric that FAISSRetriever requires."
            )
        if index.ntotal != len(chunks):
            raise RetrieverError(
                f"'{directory}' is inconsistent: {len(chunks)} chunks "
                f"but {index.ntotal} vectors."
            )

        retriever = cls(m=m, ef_construction=ef_construction, ef_search=ef_search)
        retriever._index = index
        retriever._dimension = dimension
        retriever._chunks = chunks
        return retriever

    def __len__(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)
