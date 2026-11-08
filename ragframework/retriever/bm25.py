"""BM25 lexical retrieval backed by rank_bm25."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import rank_bm25
except ImportError:
    _BM25_AVAILABLE = False
else:
    _BM25_AVAILABLE = True

from ragframework.base import Chunk, Retriever
from ragframework.exceptions import RetrieverError

logger = logging.getLogger(__name__)
_INSTALL_HINT = (
    "BM25 support requires 'ragframework[bm25]'. " "Install it with: pip install ragframework[bm25]"
)
_VARIANTS: dict[str, Any] = (
    {
        "okapi": rank_bm25.BM25Okapi,
        "l": rank_bm25.BM25L,
        "plus": rank_bm25.BM25Plus,
    }
    if _BM25_AVAILABLE
    else {}
)


def _default_tokenizer(text: str) -> list[str]:
    return text.lower().split()


class BM25Retriever(Retriever):
    """Lexical retriever using BM25 via ``rank_bm25``.

    Unlike vector-based retrievers, ``add()`` accepts chunks **without
    embeddings**: BM25 scores on ``chunk.content`` only. ``retrieve()`` ignores
    ``query_embedding`` and requires ``query_text``.

    Retrieval has two steps. First, only chunks that share at least one token
    with the query are candidates, as with an inverted index. Second, the
    candidates are ranked by their rank_bm25 score. A query with no matching
    token returns an empty list, for every variant. Ties keep insertion order.

    The index is rebuilt lazily on the first ``retrieve()`` after ``add()``,
    so bulk ingestion costs a single rebuild. Not thread-safe: call ``add()``
    before serving queries.

    Args:
        variant: ``"l"`` (default), ``"okapi"`` or ``"plus"`` (case-insensitive).
        tokenizer: Callable mapping text to tokens, used for both chunks and
            queries. Defaults to lowercase + whitespace split, so punctuation
            stays attached to tokens (``"timeout."`` != ``"timeout"``).
        k1: Term-frequency saturation parameter.
        b: Length-normalisation parameter.

    Raises:
        ImportError: If the optional ``rank_bm25`` dependency is unavailable.
        RetrieverError: If a constructor argument is invalid.
    """

    def __init__(
        self,
        variant: str = "l",
        tokenizer: Callable[[str], list[str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not _BM25_AVAILABLE:
            raise ImportError(_INSTALL_HINT)
        if not isinstance(variant, str) or variant.lower() not in _VARIANTS:
            raise RetrieverError(
                f"Invalid BM25 variant {variant!r}. Available: {', '.join(_VARIANTS)}"
            )
        if tokenizer is not None and not callable(tokenizer):
            raise RetrieverError("tokenizer must be callable.")
        if (
            not isinstance(k1, (int | float))
            or isinstance(k1, bool)
            or not math.isfinite(k1)
            or k1 <= 0
        ):
            raise RetrieverError("k1 must be a number greater than 0.")
        if (
            not isinstance(b, (int | float))
            or isinstance(b, bool)
            or not math.isfinite(b)
            or not 0 <= b <= 1
        ):
            raise RetrieverError("b must be a number between 0 and 1.")

        self.variant = variant.lower()
        self._bm25_cls = _VARIANTS[self.variant]
        self.tokenizer = tokenizer or _default_tokenizer
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._token_sets: list[set[str]] = []
        self._index: Any | None = None
        self._dirty = True

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the corpus. Embeddings are not required.

        An empty batch is a no-op. Every chunk is validated before the corpus
        is changed.
        """
        if not chunks:
            return
        for chunk in chunks:
            if not isinstance(chunk.content, str):
                raise RetrieverError(f"Chunk '{chunk.id}' has no text content.")

        self._chunks.extend(chunks)
        self._dirty = True
        logger.debug("Added chunks count=%d", len(chunks))

    def _build_index(self) -> None:
        """Rebuild the BM25 index and per-chunk token sets if the corpus changed."""
        if not self._dirty:
            return
        tokenized = [self.tokenizer(c.content) for c in self._chunks]
        self._token_sets = [set(tokens) for tokens in tokenized]
        # rank_bm25 divides by the average document length, so it cannot be
        # built when every chunk tokenizes to nothing.
        self._index = self._bm25_cls(tokenized, k1=self.k1, b=self.b) if any(tokenized) else None
        self._dirty = False

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        *,
        query_text: str | None = None,
    ) -> list[Chunk]:
        """Return up to ``top_k`` chunks that contain a query token, by BM25 score.

        Non-positive ``top_k`` values, an empty corpus and queries with no
        matching token return an empty list.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool):
            raise RetrieverError("top_k must be an integer.")
        if query_text is None:
            raise RetrieverError("BM25Retriever requires query_text to perform retrieval.")
        if not isinstance(query_text, str):
            raise RetrieverError("query_text must be a string.")
        if top_k <= 0 or not self._chunks:
            return []

        self._build_index()
        tokens = self.tokenizer(query_text)
        if self._index is None or not tokens:
            return []

        query_set = set(tokens)
        scores = self._index.get_scores(tokens)
        order = np.argsort(-scores, kind="stable")
        results = [self._chunks[int(i)] for i in order if self._token_sets[int(i)] & query_set]
        return results[:top_k]

    def __len__(self) -> int:
        """Return the number of chunks in the corpus."""
        return len(self._chunks)

    def clear(self) -> None:
        """Remove all chunks and reset the lazy BM25 index."""
        self._chunks = []
        self._token_sets = []
        self._index = None
        self._dirty = True
