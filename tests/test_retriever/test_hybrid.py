"""Tests for HybridRetriever."""

import pytest

from ragframework.base import Chunk, Retriever
from ragframework.exceptions import RetrieverError
from ragframework.retriever.hybrid import HybridRetriever


class StubRetriever(Retriever):
    def __init__(self, results: list[Chunk] | None = None) -> None:
        self.results = results or []
        self.added: list[list[Chunk]] = []
        self.calls: list[tuple[list[float], int, str | None]] = []

    def add(self, chunks: list[Chunk]) -> None:
        self.added.append(chunks)

    def retrieve(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        *,
        query_text: str | None = None,
    ) -> list[Chunk]:
        self.calls.append((query_embedding, top_k, query_text))
        return self.results[:top_k]

    def __len__(self) -> int:
        return len(self.results)

    def clear(self) -> None:
        self.results = []
        self.added.clear()
        self.calls.clear()


class IngestingStubRetriever(StubRetriever):
    def add(self, chunks: list[Chunk]) -> None:
        super().add(chunks)
        self.results.extend(chunks)


def chunk(chunk_id: str) -> Chunk:
    return Chunk(id=chunk_id, content=f"content {chunk_id}")


def test_subclasses_retriever() -> None:
    assert issubclass(HybridRetriever, Retriever)


@pytest.mark.parametrize("k", [0, -1, True, "60"])
def test_invalid_k_raises(k: object) -> None:
    with pytest.raises(RetrieverError, match="k must be a positive integer"):
        HybridRetriever(StubRetriever(), StubRetriever(), k=k)  # type: ignore[arg-type]


def test_add_forwards_chunks_to_both_retrievers() -> None:
    dense = StubRetriever()
    sparse = StubRetriever()
    retriever = HybridRetriever(dense, sparse)
    chunks = [chunk("one"), chunk("two")]

    retriever.add(chunks)

    assert dense.added == [chunks]
    assert sparse.added == [chunks]


def test_empty_add_is_a_noop() -> None:
    dense = StubRetriever()
    sparse = StubRetriever()
    HybridRetriever(dense, sparse).add([])

    assert dense.added == []
    assert sparse.added == []


def test_rrf_ranks_chunk_found_by_both_first() -> None:
    shared = chunk("shared")
    dense_only = chunk("dense-only")
    sparse_only = chunk("sparse-only")
    dense = StubRetriever([shared, dense_only])
    sparse = StubRetriever([shared, sparse_only])

    results = HybridRetriever(dense, sparse).retrieve([1.0, 0.0], top_k=3, query_text="shared")

    assert [item.id for item in results] == ["shared", "dense-only", "sparse-only"]


def test_retrieve_requests_twice_top_k_and_forwards_query() -> None:
    dense = StubRetriever([chunk("dense")])
    sparse = StubRetriever([chunk("sparse")])
    retriever = HybridRetriever(dense, sparse)

    retriever.retrieve([1.0, 2.0], top_k=4, query_text="the query")

    assert dense.calls == [([1.0, 2.0], 8, "the query")]
    assert sparse.calls == [([1.0, 2.0], 8, "the query")]


def test_duplicate_ids_are_deduplicated() -> None:
    dense = StubRetriever([chunk("same")])
    sparse = StubRetriever([Chunk(id="same", content="different object")])

    results = HybridRetriever(dense, sparse).retrieve([1.0], top_k=5, query_text="same")

    assert len(results) == 1
    assert results[0] is dense.results[0]


def test_duplicate_ids_within_one_ranking_are_scored_once() -> None:
    a = chunk("a")
    b = chunk("b")
    dense = StubRetriever([a, a, a, b])
    sparse = StubRetriever([b])

    results = HybridRetriever(dense, sparse).retrieve([1.0], top_k=2, query_text="query")
    print(results)
    assert [item.id for item in results] == ["b", "a"]


def test_repeated_ingestion_does_not_inflate_duplicate_scores() -> None:
    a = chunk("a")
    b = chunk("b")
    dense = IngestingStubRetriever([a])
    sparse = StubRetriever([b])
    retriever = HybridRetriever(dense, sparse)

    dense.add([a])
    dense.add([a])
    dense.add([b])

    results = retriever.retrieve([1.0], top_k=2, query_text="query")

    assert [item.id for item in results] == ["b", "a"]


def test_empty_child_results_degrade_to_other_ranking() -> None:
    sparse_result = chunk("sparse")
    results = HybridRetriever(
        StubRetriever(),
        StubRetriever([sparse_result]),
    ).retrieve([1.0], top_k=2, query_text="query")

    assert results == [sparse_result]


@pytest.mark.parametrize("top_k", ["2", True])
def test_invalid_top_k_raises(top_k: object) -> None:
    retriever = HybridRetriever(StubRetriever(), StubRetriever())

    with pytest.raises(RetrieverError, match="top_k must be an integer"):
        retriever.retrieve([1.0], top_k=top_k, query_text="query")  # type: ignore[arg-type]


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_returns_empty_without_querying_children(top_k: int) -> None:
    dense = StubRetriever([chunk("dense")])
    sparse = StubRetriever([chunk("sparse")])

    results = HybridRetriever(dense, sparse).retrieve([1.0], top_k=top_k, query_text="query")

    assert results == []
    assert dense.calls == []
    assert sparse.calls == []


def test_missing_query_text_raises_before_querying_children() -> None:
    dense = StubRetriever()
    sparse = StubRetriever()

    with pytest.raises(RetrieverError, match="requires query_text"):
        HybridRetriever(dense, sparse).retrieve([1.0])

    assert dense.calls == []
    assert sparse.calls == []


def test_non_string_query_text_raises_before_querying_children() -> None:
    dense = StubRetriever()
    sparse = StubRetriever()

    with pytest.raises(RetrieverError, match="query_text must be a string"):
        HybridRetriever(dense, sparse).retrieve([1.0], query_text=123)  # type: ignore[arg-type]

    assert dense.calls == []
    assert sparse.calls == []
