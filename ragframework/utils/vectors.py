"""Shared validation and normalization for retriever vectors."""

from __future__ import annotations

from typing import Any

import numpy as np

from ragframework.exceptions import RetrieverError


def validate_vector(values: list[float], label: str) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Return a contiguous, L2-normalized float32 vector.

    Raise ``RetrieverError`` with the supplied label for non-numeric, empty,
    multidimensional, non-finite, or zero-norm vectors.
    """
    try:
        vector = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise RetrieverError(f"{label} must contain only numeric values.") from exc
    if vector.ndim != 1 or vector.size == 0:
        raise RetrieverError(f"{label} must be a non-empty one-dimensional vector.")
    if not np.isfinite(vector).all():
        raise RetrieverError(f"{label} must contain only finite values.")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise RetrieverError(f"{label} must not be a zero vector.")
    return np.ascontiguousarray(vector / norm, dtype=np.float32)
