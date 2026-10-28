"""Tests for ChromaRetriever."""

import json
import logging
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


def test_retrieve_filters_by_metadata():
    retriever = ChromaRetriever(collection_name="test_metadata_filter")
    retriever.add(
        [
            make_chunk(
                "north",
                [1.0, 0.0],
                {"region": "north", "active": True},
            ),
            make_chunk(
                "south",
                [0.0, 1.0],
                {"region": "south", "active": True},
            ),
            make_chunk(
                "inactive",
                [0.8, 0.2],
                {"region": "north", "active": False},
            ),
        ]
    )

    results = retriever.retrieve(
        [1.0, 0.0],
        top_k=3,
        filter={"region": "north", "active": True},
    )

    assert [chunk.id for chunk in results] == ["north"]


def test_retrieve_filter_with_no_matches_returns_empty():
    retriever = ChromaRetriever(collection_name="test_metadata_filter_no_match")
    retriever.add([make_chunk("north", [1.0, 0.0], {"region": "north"})])

    assert (
        retriever.retrieve(
            [1.0, 0.0],
            filter={"region": "missing"},
        )
        == []
    )


@pytest.mark.parametrize(
    "filter_value",
    [["north"], {"nested": "value"}, None],
)
def test_retrieve_rejects_non_scalar_filter_values(filter_value):
    retriever = ChromaRetriever(collection_name="test_metadata_filter_invalid")
    retriever.add([make_chunk("north", [1.0, 0.0], {"region": "north"})])

    with pytest.raises(RetrieverError):
        retriever.retrieve(
            [1.0, 0.0],
            filter={"region": filter_value},
        )


def test_add_logs_count_and_dimension(caplog):
    retriever = ChromaRetriever(
        collection_name="test_add_logging",
    )
    chunks = [
        make_chunk("log-1", [1.0, 0.0, 0.0]),
        make_chunk("log-2", [0.0, 1.0, 0.0]),
    ]

    with caplog.at_level(
        logging.DEBUG,
        logger="ragframework.retriever.chroma",
    ):
        retriever.add(chunks)

    assert "Added chunks count=2 dimension=3" in caplog.text


def test_clear_empties_collection_and_allows_readding():
    retriever = ChromaRetriever(collection_name="test_clear")
    retriever.add([make_chunk("old", [1.0, 0.0])])
    retriever.clear()
    assert len(retriever) == 0
    assert retriever.retrieve([1.0, 0.0]) == []
    retriever.add([make_chunk("new", [1.0, 0.0])])
    assert [chunk.id for chunk in retriever.retrieve([1.0, 0.0])] == ["new"]


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


def test_metadata_preserves_scalar_values():
    chunk = Chunk(
        id="metadata-test",
        content="Test content",
        embedding=[1.0, 0.0],
        metadata={"category": "science", "count": 3, "active": True},
    )

    stored = ChromaRetriever._metadata(chunk)

    assert stored["category"] == "science"
    assert stored["count"] == 3
    assert stored["active"] is True
    assert json.loads(stored["_ragframework_metadata"]) == chunk.metadata


def test_restore_metadata_without_serialized_value():
    metadata = {"category": "science", "count": 3}

    assert ChromaRetriever._restore_metadata(metadata) == metadata


def test_restore_metadata_with_invalid_json():
    metadata = {"_ragframework_metadata": "{invalid-json", "category": "science"}

    assert ChromaRetriever._restore_metadata(metadata) == metadata


def test_restore_metadata_with_non_dictionary_json():
    metadata = {"_ragframework_metadata": '["science", "history"]'}

    assert ChromaRetriever._restore_metadata(metadata) == metadata


def test_add_chunk_without_embedding_raises_error():
    from ragframework.base import Chunk
    from ragframework.exceptions import RetrieverError
    from ragframework.retriever.chroma import ChromaRetriever

    retriever = ChromaRetriever()
    chunk = Chunk(
        id="missing-embedding",
        content="Test content",
        embedding=None,
        metadata={},
    )

    with pytest.raises(RetrieverError, match="has no embedding"):
        retriever.add([chunk])


def test_retrieve_rejects_non_finite_filter():
    from ragframework.exceptions import RetrieverError
    from ragframework.retriever.chroma import ChromaRetriever

    retriever = ChromaRetriever()

    with pytest.raises(RetrieverError, match="must be finite"):
        retriever.retrieve([1.0, 0.0], filter={"score": float("nan")})


def test_retrieve_wraps_query_exception():
    retriever = ChromaRetriever(collection_name="test_query_error")
    retriever.add([make_chunk("query-error", [1.0, 0.0])])

    def raise_query_error(**kwargs):
        raise RuntimeError("simulated query failure")

    retriever._collection.query = raise_query_error

    with pytest.raises(RetrieverError, match="Failed to query ChromaDB"):
        retriever.retrieve([1.0, 0.0])


def test_clear_wraps_delete_exception():
    retriever = ChromaRetriever(collection_name="test_clear_error")

    def raise_delete_error(**kwargs):
        raise RuntimeError("simulated delete failure")

    retriever._client.delete_collection = raise_delete_error

    with pytest.raises(RetrieverError, match="Failed to clear ChromaDB collection"):
        retriever.clear()


def test_retrieve_handles_empty_result_groups():
    retriever = ChromaRetriever(collection_name="test_empty_result_groups")
    retriever.add([make_chunk("empty-groups", [1.0, 0.0])])

    retriever._collection.query = lambda **kwargs: {
        "ids": [],
        "documents": [],
        "metadatas": [],
    }

    assert retriever.retrieve([1.0, 0.0]) == []
