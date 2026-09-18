"""Tests for ChromaRetriever."""

from pathlib import Path

import pytest

from ragframework.base import Chunk
from ragframework.exceptions import RetrieverError
from ragframework.retriever import ChromaRetriever


def make_chunk(
    chunk_id: str,
    embedding: list[float],
    metadata: dict | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        content=f"Content {chunk_id}",
        metadata=metadata or {},
        embedding=embedding,
    )


def test_empty_retriever_returns_empty():
    retriever = ChromaRetriever(
        collection_name="test_empty_retriever",
    )

    assert retriever.retrieve([1.0, 0.0, 0.0], top_k=3) == []
    assert len(retriever) == 0


def test_add_and_retrieve():
    retriever = ChromaRetriever(
        collection_name="test_add_and_retrieve",
    )

    chunks = [
        make_chunk(
            "chunk-1",
            [1.0, 0.0, 0.0],
            {"source": "test.txt"},
        ),
        make_chunk(
            "chunk-2",
            [0.0, 1.0, 0.0],
            {"source": "test.txt"},
        ),
    ]

    retriever.add(chunks)

    results = retriever.retrieve(
        [1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].id == "chunk-1"
    assert results[0].content == "Content chunk-1"
    assert results[0].metadata["source"] == "test.txt"


def test_retrieve_top_k_limited():
    retriever = ChromaRetriever(
        collection_name="test_top_k",
    )

    chunks = [
        make_chunk("chunk-1", [1.0, 0.0, 0.0]),
        make_chunk("chunk-2", [0.0, 1.0, 0.0]),
        make_chunk("chunk-3", [0.0, 0.0, 1.0]),
    ]

    retriever.add(chunks)

    results = retriever.retrieve(
        [1.0, 0.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2


def test_retrieve_returns_most_similar_first():
    retriever = ChromaRetriever(
        collection_name="test_similarity",
    )

    chunks = [
        make_chunk("a", [1.0, 0.0, 0.0]),
        make_chunk("b", [0.0, 1.0, 0.0]),
        make_chunk("c", [0.0, 0.0, 1.0]),
    ]

    retriever.add(chunks)

    results = retriever.retrieve(
        [1.0, 0.0, 0.0],
        top_k=3,
    )

    assert results[0].id == "a"


def test_chunk_without_embedding_raises():
    retriever = ChromaRetriever(
        collection_name="test_missing_embedding",
    )

    chunk = Chunk(
        id="bad",
        content="No embedding",
    )

    with pytest.raises(
        RetrieverError,
        match="has no embedding",
    ):
        retriever.add([chunk])


def test_empty_chunks_are_ignored():
    retriever = ChromaRetriever(
        collection_name="test_empty_chunks",
    )

    retriever.add([])

    assert len(retriever) == 0


def test_non_positive_top_k_returns_empty():
    retriever = ChromaRetriever(
        collection_name="test_non_positive_top_k",
    )

    retriever.add(
        [
            make_chunk("chunk-1", [1.0, 0.0]),
        ]
    )

    assert (
        retriever.retrieve(
            [1.0, 0.0],
            top_k=0,
        )
        == []
    )

    assert (
        retriever.retrieve(
            [1.0, 0.0],
            top_k=-1,
        )
        == []
    )


def test_metadata_is_preserved():
    retriever = ChromaRetriever(
        collection_name="test_metadata",
    )

    metadata = {
        "source": "document.pdf",
        "page": 3,
        "important": True,
    }

    retriever.add(
        [
            make_chunk(
                "metadata-test",
                [1.0, 0.0],
                metadata,
            ),
        ]
    )

    results = retriever.retrieve(
        [1.0, 0.0],
        top_k=1,
    )

    assert results[0].metadata == metadata


def test_empty_metadata_is_restored_as_empty_dict():
    retriever = ChromaRetriever(
        collection_name="test_empty_metadata",
    )

    retriever.add(
        [
            make_chunk(
                "empty-metadata",
                [1.0, 0.0],
            ),
        ]
    )

    results = retriever.retrieve(
        [1.0, 0.0],
        top_k=1,
    )

    assert results[0].metadata == {}


def test_metadata_values_are_converted_to_strings_when_needed():
    retriever = ChromaRetriever(
        collection_name="test_metadata_conversion",
    )

    retriever.add(
        [
            make_chunk(
                "metadata-conversion",
                [1.0, 0.0],
                {
                    "source": "test",
                    "tags": ["rag", "llm"],
                },
            ),
        ]
    )

    results = retriever.retrieve(
        [1.0, 0.0],
        top_k=1,
    )

    assert results[0].metadata["source"] == "test"
    assert results[0].metadata["tags"] == "['rag', 'llm']"


def test_persistent_storage(tmp_path: Path):
    persist_directory = tmp_path / "chroma"

    first_retriever = ChromaRetriever(
        collection_name="test_persistence",
        persist_directory=str(persist_directory),
    )

    first_retriever.add(
        [
            make_chunk(
                "persistent-chunk",
                [1.0, 0.0, 0.0],
                {"source": "persistent.txt"},
            ),
        ]
    )

    assert len(first_retriever) == 1

    second_retriever = ChromaRetriever(
        collection_name="test_persistence",
        persist_directory=str(persist_directory),
    )

    assert len(second_retriever) == 1

    results = second_retriever.retrieve(
        [1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].id == "persistent-chunk"
    assert results[0].metadata["source"] == "persistent.txt"


def test_user_metadata_with_old_empty_marker_is_preserved():
    retriever = ChromaRetriever(
        collection_name="test_metadata_marker_collision",
    )

    metadata = {
        "_ragframework_empty_metadata": True,
        "source": "document.pdf",
        "page": 3,
    }

    retriever.add(
        [
            make_chunk(
                "collision-test",
                [1.0, 0.0],
                metadata,
            )
        ]
    )

    results = retriever.retrieve(
        [1.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].metadata == metadata
