"""Tests for InMemoryRetriever."""

import logging

import pytest

from ragframework.base import Chunk
from ragframework.exceptions import RetrieverError
from ragframework.retriever.in_memory import InMemoryRetriever


def make_chunk(cid: str, embedding: list[float]) -> Chunk:
    return Chunk(id=cid, content=f"Content {cid}", embedding=embedding)


class TestInMemoryRetriever:
    def test_empty_retriever_returns_empty(self):
        r = InMemoryRetriever()
        result = r.retrieve([0.1, 0.2, 0.3], top_k=3)
        assert result == []

    def test_add_and_retrieve(self, embedded_chunks):
        r = InMemoryRetriever()
        r.add(embedded_chunks)
        assert len(r) == len(embedded_chunks)

    def test_add_logs_count_and_dimension(self, embedded_chunks, caplog):
        retriever = InMemoryRetriever()
        chunks = embedded_chunks[:2]

        with caplog.at_level(
            logging.DEBUG,
            logger="ragframework.retriever.in_memory",
        ):
            retriever.add(chunks)

        assert "Added chunks count=2 dimension=16" in caplog.text

    def test_retrieve_top_k_limited(self, embedded_chunks):
        r = InMemoryRetriever()
        r.add(embedded_chunks)
        query = embedded_chunks[0].embedding
        results = r.retrieve(query, top_k=2)
        assert len(results) <= 2

    def test_chunk_without_embedding_raises(self):
        r = InMemoryRetriever()
        bad_chunk = Chunk(id="bad", content="no embedding")
        with pytest.raises(RetrieverError, match="no embedding"):
            r.add([bad_chunk])

    def test_retrieve_returns_most_similar_first(self):
        r = InMemoryRetriever()
        # chunk_a is identical to the query — highest similarity
        chunk_a = make_chunk("a", [1.0, 0.0, 0.0, 0.0])
        chunk_b = make_chunk("b", [0.0, 1.0, 0.0, 0.0])
        chunk_c = make_chunk("c", [0.0, 0.0, 1.0, 0.0])
        r.add([chunk_a, chunk_b, chunk_c])
        results = r.retrieve([1.0, 0.0, 0.0, 0.0], top_k=3)
        assert results[0].id == "a"

    @pytest.mark.parametrize("populated", [False, True])
    def test_empty_batch_is_a_no_op(self, populated):
        r = InMemoryRetriever()
        existing = make_chunk("existing", [1.0, 0.0])
        if populated:
            r.add([existing])

        r.add([])

        assert len(r) == int(populated)
        assert r.retrieve([1.0, 0.0]) == ([existing] if populated else [])

    @pytest.mark.parametrize("populated", [False, True])
    @pytest.mark.parametrize(
        ("embedding", "message"),
        [
            (None, "no embedding"),
            ([], "non-empty one-dimensional"),
            ([[1.0, 0.0]], "non-empty one-dimensional"),
            (["not-a-number", 0.0], "numeric"),
            ([float("nan"), 0.0], "finite"),
            ([float("inf"), 0.0], "finite"),
            ([0.0, 0.0], "zero vector"),
        ],
    )
    def test_invalid_batch_leaves_retriever_usable(self, populated, embedding, message):
        r = InMemoryRetriever()
        existing = make_chunk("existing", [1.0, 0.0])
        if populated:
            r.add([existing])

        with pytest.raises(RetrieverError, match=message):
            r.add([make_chunk("good", [0.0, 1.0]), make_chunk("bad", embedding)])

        assert len(r) == int(populated)
        assert r.retrieve([1.0, 0.0]) == ([existing] if populated else [])
        # A failed first batch must not establish the index dimension.
        recovered = make_chunk("recovered", [0.0, 2.0] if populated else [0.0, 2.0, 0.0])
        r.add([recovered])
        assert r.retrieve(recovered.embedding, top_k=1) == [recovered]

    def test_inconsistent_first_batch_does_not_establish_dimension(self):
        r = InMemoryRetriever()
        with pytest.raises(
            RetrieverError, match="Chunk 'three' embedding has dimension 3; expected 2"
        ):
            r.add([make_chunk("two", [1.0, 0.0]), make_chunk("three", [1.0, 0.0, 0.0])])

        assert len(r) == 0
        assert r.retrieve([1.0, 0.0]) == []
        recovered = make_chunk("recovered", [1.0, 0.0, 0.0])
        r.add([recovered])
        assert r.retrieve([1.0, 0.0, 0.0]) == [recovered]

    def test_wrong_dimension_append_preserves_existing_results(self):
        r = InMemoryRetriever()
        north = make_chunk("north", [10.0, 0.0])
        east = make_chunk("east", [0.0, 3.0])
        r.add([east, north])
        with pytest.raises(
            RetrieverError, match="Chunk 'bad' embedding has dimension 3; expected 2"
        ):
            r.add([make_chunk("good", [1.0, 1.0]), make_chunk("bad", [1.0, 0.0, 0.0])])

        assert len(r) == 2
        assert r.retrieve([2.0, 0.0]) == [north, east]
        northeast = make_chunk("northeast", [1.0, 1.0])
        r.add([northeast])
        assert len(r) == 3
        assert r.retrieve([2.0, 0.0]) == [north, northeast, east]

    @pytest.mark.parametrize(
        ("embedding", "message"),
        [
            ([], "non-empty one-dimensional"),
            ([[1.0, 0.0]], "non-empty one-dimensional"),
            (["not-a-number", 0.0], "numeric"),
            ([float("nan"), 0.0], "finite"),
            ([float("inf"), 0.0], "finite"),
            ([0.0, 0.0], "zero vector"),
            ([1.0, 0.0, 0.0], "has dimension 3; expected 2"),
        ],
    )
    def test_invalid_query_raises_retriever_error(self, embedding, message):
        r = InMemoryRetriever()
        existing = make_chunk("existing", [1.0, 0.0])
        r.add([existing])
        with pytest.raises(RetrieverError, match=f"Query embedding .*{message}"):
            r.retrieve(embedding)

        assert r.retrieve([1.0, 0.0]) == [existing]

    def test_non_positive_top_k_returns_empty(self):
        r = InMemoryRetriever()
        r.add(
            [
                make_chunk("a", [1.0, 0.0]),
                make_chunk("b", [0.0, 1.0]),
                make_chunk("c", [0.5, 0.5]),
                make_chunk("d", [1.0, 1.0]),
                make_chunk("e", [0.2, 0.8]),
            ]
        )

        assert r.retrieve([1.0, 0.0], top_k=0) == []
        assert r.retrieve([1.0, 0.0], top_k=-1) == []

    def test_bool_top_k_raises_retriever_error(self):
        r = InMemoryRetriever()
        r.add([make_chunk("a", [1.0, 0.0])])

        with pytest.raises(RetrieverError, match="top_k must be an integer"):
            r.retrieve([1.0, 0.0], top_k=True)

    def test_batched_add_matches_single_batch(self):
        chunks = [
            make_chunk("a", [1.0, 0.0]),
            make_chunk("b", [0.0, 1.0]),
            make_chunk("c", [1.0, 1.0]),
            make_chunk("d", [-1.0, 0.0]),
        ]

        single_batch = InMemoryRetriever()
        single_batch.add(chunks)

        batched = InMemoryRetriever()
        batched.add(chunks[:2])
        batched.add(chunks[2:])

        query = [1.0, 1.0]

        single_results = single_batch.retrieve(query, top_k=4)
        batched_results = batched.retrieve(query, top_k=4)

        assert len(batched_results) == len(single_results)
        assert batched_results[0].id == single_results[0].id

def test_retrieve_consolidates_matrix_blocks_without_duplicate_storage():
    retriever = InMemoryRetriever()

    chunks = [
        make_chunk("a", [1.0, 0.0]),
        make_chunk("b", [0.0, 1.0]),
        make_chunk("c", [1.0, 1.0]),
    ]

    retriever.add(chunks[:1])
    retriever.add(chunks[1:])

    assert len(retriever._matrix_blocks) == 2
    assert retriever._matrix is None

    retriever.retrieve([1.0, 1.0], top_k=3)

    assert retriever._matrix is not None
    assert len(retriever._matrix_blocks) == 1
    assert retriever._matrix_blocks[0] is retriever._matrix

def test_add_after_consolidation_preserves_all_results():
    retriever = InMemoryRetriever()

    first = [
        make_chunk("a", [1.0, 0.0]),
        make_chunk("b", [0.0, 1.0]),
    ]
    second = [
        make_chunk("c", [1.0, 1.0]),
    ]

    retriever.add(first)
    retriever.retrieve([1.0, 1.0], top_k=2)

    assert len(retriever._matrix_blocks) == 1
    assert retriever._matrix_blocks[0] is retriever._matrix

    retriever.add(second)

    assert retriever._matrix is None
    assert len(retriever._matrix_blocks) == 2

    results = retriever.retrieve([1.0, 1.0], top_k=3)

    assert [chunk.id for chunk in results] == ["c", "b", "a"]
    assert len(retriever._matrix_blocks) == 1
    assert retriever._matrix_blocks[0] is retriever._matrix
