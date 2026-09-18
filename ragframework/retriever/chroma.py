"""ChromaDB-backed retriever."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from ragframework.base import Chunk, Retriever
from ragframework.exceptions import RetrieverError


class ChromaRetriever(Retriever):
    """Retriever backed by ChromaDB.

    Supports both ephemeral in-memory storage and persistent on-disk storage.

    Args:
        collection_name: Name of the ChromaDB collection.
        persist_directory: Directory for persistent storage. If ``None``,
            ChromaDB uses an ephemeral in-memory client.
    """

    _METADATA_KEY = "_ragframework_metadata"

    def __init__(
        self,
        collection_name: str = "ragframework",
        persist_directory: str | None = None,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise ImportError(
                "ChromaDB support requires 'ragframework[chromadb]'. "
                "Install it with: pip install ragframework[chromadb]"
            ) from exc

        try:
            if persist_directory is None:
                self._client = chromadb.Client()
            else:
                self._client = chromadb.PersistentClient(
                    path=persist_directory,
                )

            self._collection = self._client.get_or_create_collection(
                name=collection_name,
            )
        except Exception as exc:
            raise RetrieverError(f"Could not initialize ChromaDB: {exc}") from exc

    def add(self, chunks: list[Chunk]) -> None:
        """Add embedded chunks to the ChromaDB collection."""
        if not chunks:
            return

        embeddings: list[Sequence[float]] = []

        for chunk in chunks:
            if chunk.embedding is None:
                raise RetrieverError(
                    f"Chunk '{chunk.id}' has no embedding. "
                    "Embed chunks before adding them to the retriever."
                )

            embeddings.append(chunk.embedding)

        try:
            self._collection.upsert(
                ids=[chunk.id for chunk in chunks],
                embeddings=embeddings,
                documents=[chunk.content for chunk in chunks],
                metadatas=[self._metadata(chunk) for chunk in chunks],
            )
        except Exception as exc:
            raise RetrieverError(f"Failed to add chunks to ChromaDB: {exc}") from exc

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[Chunk]:
        """Return the most similar chunks for a query embedding."""
        if top_k <= 0:
            return []

        query_embeddings: list[Sequence[float]] = [query_embedding]

        try:
            result = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise RetrieverError(f"Failed to query ChromaDB: {exc}") from exc

        ids_result = result.get("ids") or []
        documents_result = result.get("documents") or []
        metadatas_result = result.get("metadatas") or []

        ids = ids_result[0] if ids_result else []
        documents = documents_result[0] if documents_result else []
        metadatas = metadatas_result[0] if metadatas_result else []

        chunks_result: list[Chunk] = []

        for chunk_id, document, metadata in zip(
            ids,
            documents,
            metadatas,
            strict=True,
        ):
            chunks_result.append(
                Chunk(
                    id=chunk_id,
                    content=document or "",
                    metadata=self._restore_metadata(metadata or {}),
                )
            )

        return chunks_result

    @classmethod
    def _metadata(cls, chunk: Chunk) -> dict[str, str]:
        """Serialize chunk metadata into a collision-safe Chroma value."""
        metadata: dict[str, Any] = {}

        for key, value in chunk.metadata.items():
            if isinstance(value, str | int | float | bool):
                metadata[key] = value
            else:
                metadata[key] = str(value)

        return {
            cls._METADATA_KEY: json.dumps(
                metadata,
                sort_keys=True,
            )
        }

    @classmethod
    def _restore_metadata(
        cls,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Restore framework metadata from ChromaDB."""
        serialized = metadata.get(cls._METADATA_KEY)

        if not isinstance(serialized, str):
            return dict(metadata)

        try:
            restored = json.loads(serialized)
        except json.JSONDecodeError:
            return dict(metadata)

        if isinstance(restored, dict):
            return restored

        return dict(metadata)

    def __len__(self) -> int:
        """Return the number of chunks stored in the collection."""
        return int(self._collection.count())
