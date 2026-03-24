"""Placeholder embedder for testing without an API key."""

from __future__ import annotations

import random

from ragframework.base import Embedder


class RandomEmbedder(Embedder):
    """Produce random unit-normalised vectors — for testing only.

    This embedder lets you run the full pipeline end-to-end without any
    external service. Retrieval results will be meaningless, but the
    pipeline plumbing will work correctly.

    Replace this with :class:`OpenAIEmbedder` or
    :class:`HuggingFaceEmbedder` for real use-cases (see
    ``.github/GOOD_FIRST_ISSUES.md``).

    Args:
        dim: Dimensionality of the produced vectors (default: 384).
        seed: Optional RNG seed for reproducible results in tests.
    """

    def __init__(self, dim: int = 384, seed: int | None = None) -> None:
        self.dim = dim
        self._rng = random.Random(seed)

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for _ in texts:
            vec = [self._rng.gauss(0, 1) for _ in range(self.dim)]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            results.append([v / norm for v in vec])
        return results
