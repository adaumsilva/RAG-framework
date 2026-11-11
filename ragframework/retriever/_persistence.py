"""Validation helpers shared by the retriever ``save()`` / ``load()`` methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragframework.base import Chunk
from ragframework.exceptions import RetrieverError


def require_json_object(value: object, filename: str, directory: Path) -> dict[str, Any]:
    """Return *value* if it is a JSON object, otherwise raise ``RetrieverError``."""
    if not isinstance(value, dict):
        raise RetrieverError(
            f"Malformed {filename} in '{directory}': expected a JSON object, "
            f"got {type(value).__name__}."
        )
    return value


def check_format_version(found: object, expected: int, filename: str, directory: Path) -> None:
    """Raise ``RetrieverError`` if the stored format version is not *expected*."""
    if isinstance(found, bool) or found != expected:
        raise RetrieverError(
            f"'{directory}' uses index format {found!r} (from {filename}), "
            f"but this version of ragframework reads format {expected}. "
            "Re-create the index with save()."
        )


def require_positive_int(meta: dict[str, Any], key: str, directory: Path) -> int:
    """Return ``meta[key]`` if it is a positive integer, otherwise raise ``RetrieverError``."""
    value = meta.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RetrieverError(
            f"Malformed meta.json in '{directory}': '{key}' must be a positive integer, "
            f"got {value!r}."
        )
    return value


def parse_chunk_records(records: object, directory: Path) -> list[Chunk]:
    """Convert stored chunk records to :class:`Chunk` objects without embeddings.

    Each record must be a JSON object with a string ``id``, a string
    ``content``, and an object ``metadata``.

    Raises:
        RetrieverError: If *records* is not a list or any record is malformed.
    """
    if not isinstance(records, list):
        raise RetrieverError(
            f"Malformed chunks.json in '{directory}': expected a list of chunk records, "
            f"got {type(records).__name__}."
        )

    chunks: list[Chunk] = []
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise RetrieverError(
                f"Malformed chunks.json in '{directory}': record {position} "
                f"must be an object, got {type(record).__name__}."
            )
        for key, expected_type in (("id", str), ("content", str), ("metadata", dict)):
            if not isinstance(record.get(key), expected_type):
                raise RetrieverError(
                    f"Malformed chunks.json in '{directory}': record {position} "
                    f"needs '{key}' of type {expected_type.__name__}."
                )
        chunks.append(
            Chunk(id=record["id"], content=record["content"], metadata=record["metadata"])
        )
    return chunks
