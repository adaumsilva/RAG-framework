"""Hybrid retrieval: fuse a dense and a sparse retriever with RRF."""

from __future__ import annotations

import logging
from collections import defaultdict

from ragframework.base import Chunk, Retriever
from ragframework.exceptions import RetrieverError

logger = logging.getLogger(__name__)


class HybridRetriever(Retriever):
    """Fuses a dense and a sparse retriever with Reciprocal Rank Fusion (RRF).

    Each child returns its top ``top_k * 2`` chunks. A chunk's fused score is
    the sum of ``1 / (k + rank)`` over the rankings it appears in (rank starts
    at 1), so chunks found by both retrievers outrank chunks found by one.
    Duplicate chunk IDs within a child ranking contribute at most once.
    Ties are broken in favour of the dense ranking.

    Chunks are matched across retrievers by ``Chunk.id``, so both children must
    be fed the same chunks with the same ids. ``add()`` guarantees this.
    ``retrieve()`` requires ``query_text`` because the sparse side needs it.

    Args:
        dense: Embedding-based retriever (e.g. ``FAISSRetriever``).
        sparse: Lexical retriever (e.g. ``BM25Retriever``).
        k: RRF smoothing constant. The original paper suggests 60.

    Raises:
        RetrieverError: If ``k`` is not a positive integer.
    """

    def __init__(self, dense: Retriever, sparse: Retriever, k: int = 60) -> None:
        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise RetrieverError("k must be a positive integer.")
        self.dense = dense
        self.sparse = sparse
        self.k = k

    def add(self, chunks: list[Chunk]) -> None:
        """Forward chunks to both retrievers. Chunks need embeddings for the dense side."""
        if not chunks:
            return
        self.dense.add(chunks)
        self.sparse.add(chunks)

    def __len__(self) -> int:
        """Return the number of chunks shared by both child retrievers."""
        return min(len(self.dense), len(self.sparse))

    def clear(self) -> None:
        """Remove all chunks from both child retrievers."""
        self.dense.clear()
        self.sparse.clear()

    def _fuse(self, rankings: list[list[Chunk]], top_k: int) -> list[Chunk]:
        """Merge ranked lists with RRF, deduplicating each ranking by chunk ID."""
        scores: dict[str, float] = defaultdict(float)
        chunks: dict[str, Chunk] = {}
        for ranking in rankings:
            seen_ids: set[str] = set()
            rank = 0
            for chunk in ranking:
                if chunk.id in seen_ids:
                    continue

                seen_ids.add(chunk.id)
                rank += 1
                chunks.setdefault(chunk.id, chunk)
                scores[chunk.id] += 1.0 / (self.k + rank)

        # sorted() is stable and dicts keep insertion order, so ties go to
        # the first ranking (dense).
        top = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
        return [chunks[chunk_id] for chunk_id in top]

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        *,
        query_text: str | None = None,
    ) -> list[Chunk]:
        """Return up to ``top_k`` chunks ordered by fused RRF score.

        Non-positive ``top_k`` values return an empty list. If one retriever
        finds nothing, the result degrades to the other's ranking.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise RetrieverError("top_k must be an integer.")
        if query_text is None:
            raise RetrieverError("HybridRetriever requires query_text to perform retrieval.")
        if not isinstance(query_text, str):
            raise RetrieverError("query_text must be a string.")
        if top_k <= 0:
            return []

        n = top_k * 2
        dense_results = self.dense.retrieve(query_embedding, n, query_text=query_text)
        sparse_results = self.sparse.retrieve(query_embedding, n, query_text=query_text)
        logger.debug(
            "Hybrid retrieve dense=%d sparse=%d",
            len(dense_results),
            len(sparse_results),
        )
        return self._fuse([dense_results, sparse_results], top_k)
