"""Tests for the FAISS-backed retriever."""

import subprocess
import sys

import faiss
import numpy as np
import pytest

from ragframework.base import Chunk, Retriever
from ragframework.exceptions import RetrieverError
from ragframework.retriever.faiss import FAISSRetriever


def make_chunk(cid: str, embedding: list[float]) -> Chunk:
    return Chunk(
        id=cid,
        content=f"Content {cid}",
        metadata={"source": cid},
        embedding=embedding,
    )


def test_subclasses_retriever() -> None:
    assert issubclass(FAISSRetriever, Retriever)


def test_real_faiss_search_uses_cosine_similarity_and_preserves_chunks() -> None:
    retriever = FAISSRetriever(ef_search=64)
    north = make_chunk("north", [10.0, 0.0])
    northeast = make_chunk("northeast", [1.0, 1.0])
    east = make_chunk("east", [0.0, 3.0])

    retriever.add([east, north, northeast])
    results = retriever.retrieve([2.0, 0.0], top_k=3)

    assert results == [north, northeast, east]
    assert results[0] is north
    assert results[0].content == "Content north"
    assert results[0].metadata == {"source": "north"}


def test_batches_append_without_replacing_existing_chunks() -> None:
    retriever = FAISSRetriever(ef_search=64)
    first = make_chunk("first", [1.0, 0.0, 0.0])
    second = make_chunk("second", [0.0, 1.0, 0.0])
    retriever.add([first])
    retriever.add([second])

    assert len(retriever) == 2
    assert retriever.retrieve([0.0, 1.0, 0.0], top_k=2)[0] is second


def test_hnsw_index_uses_configured_parameters_and_inner_product() -> None:
    retriever = FAISSRetriever(m=12, ef_construction=48, ef_search=24)
    retriever.add([make_chunk("one", [1.0, 0.0])])

    assert isinstance(retriever._index, faiss.IndexHNSWFlat)
    assert retriever._index.metric_type == faiss.METRIC_INNER_PRODUCT
    assert retriever._index.hnsw.nb_neighbors(1) == 12
    assert retriever._index.hnsw.efConstruction == 48
    assert retriever._index.hnsw.efSearch == 24


def test_empty_batch_and_empty_index_return_empty() -> None:
    retriever = FAISSRetriever()
    retriever.add([])
    assert len(retriever) == 0
    assert retriever.retrieve([1.0], top_k=3) == []


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_returns_empty(top_k: int) -> None:
    retriever = FAISSRetriever()
    retriever.add([make_chunk("one", [1.0])])
    assert retriever.retrieve([1.0], top_k=top_k) == []


def test_top_k_is_limited_to_index_size() -> None:
    retriever = FAISSRetriever()
    retriever.add([make_chunk("one", [1.0, 0.0])])
    assert retriever.retrieve([1.0, 0.0], top_k=20) == [retriever._chunks[0]]


@pytest.mark.parametrize("top_k", [1.5, True])
def test_top_k_must_be_an_integer(top_k: object) -> None:
    retriever = FAISSRetriever()
    with pytest.raises(RetrieverError, match="top_k must be an integer"):
        retriever.retrieve([1.0], top_k=top_k)  # type: ignore[arg-type]


def test_missing_embedding_does_not_partially_add_batch() -> None:
    retriever = FAISSRetriever()
    good = make_chunk("good", [1.0, 0.0])
    bad = Chunk(id="bad", content="Missing")

    with pytest.raises(RetrieverError, match="no embedding"):
        retriever.add([good, bad])

    assert len(retriever) == 0
    assert retriever._index is None


@pytest.mark.parametrize(
    ("embedding", "message"),
    [
        ([], "non-empty one-dimensional"),
        ([[1.0, 0.0]], "non-empty one-dimensional"),
        ([float("nan"), 0.0], "finite"),
        ([float("inf"), 0.0], "finite"),
        ([0.0, 0.0], "zero vector"),
    ],
)
def test_invalid_chunk_vectors_are_rejected_atomically(
    embedding: list[float], message: str
) -> None:
    retriever = FAISSRetriever()
    with pytest.raises(RetrieverError, match=message):
        retriever.add([make_chunk("good", [1.0, 0.0]), make_chunk("bad", embedding)])
    assert len(retriever) == 0


def test_inconsistent_dimensions_are_rejected_for_batch_and_append() -> None:
    retriever = FAISSRetriever()
    with pytest.raises(RetrieverError, match="expected 2"):
        retriever.add([make_chunk("two", [1.0, 0.0]), make_chunk("three", [1.0, 0.0, 0.0])])
    assert len(retriever) == 0

    retriever.add([make_chunk("two", [1.0, 0.0])])
    with pytest.raises(RetrieverError, match="expected 2"):
        retriever.add([make_chunk("three", [1.0, 0.0, 0.0])])
    assert len(retriever) == 1


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ([], "non-empty one-dimensional"),
        ([float("nan"), 0.0], "finite"),
        ([0.0, 0.0], "zero vector"),
        ([1.0, 0.0, 0.0], "expected 2"),
    ],
)
def test_invalid_queries_raise(query: list[float], message: str) -> None:
    retriever = FAISSRetriever()
    retriever.add([make_chunk("one", [1.0, 0.0])])
    with pytest.raises(RetrieverError, match=message):
        retriever.retrieve(query)


def test_invalid_faiss_labels_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = FAISSRetriever()
    chunk = make_chunk("one", [1.0, 0.0])
    retriever.add([chunk])
    monkeypatch.setattr(
        retriever._index,
        "search",
        lambda _query, _count: (
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            np.array([[0, -1, 999]], dtype=np.int64),
        ),
    )
    assert retriever.retrieve([1.0, 0.0]) == [chunk]


def test_faiss_search_failure_is_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = FAISSRetriever()
    retriever.add([make_chunk("one", [1.0, 0.0])])

    def fail(_query: np.ndarray, _count: int) -> None:
        raise RuntimeError("native failure")

    monkeypatch.setattr(retriever._index, "search", fail)
    with pytest.raises(RetrieverError, match="failed to search") as exc_info:
        retriever.retrieve([1.0, 0.0])
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_faiss_add_failure_is_wrapped_without_extending_chunk_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retriever = FAISSRetriever()
    retriever.add([make_chunk("existing", [1.0, 0.0])])

    def fail(_vectors: np.ndarray) -> None:
        raise RuntimeError("native failure")

    monkeypatch.setattr(retriever._index, "add", fail)
    with pytest.raises(RetrieverError, match="failed to add") as exc_info:
        retriever.add([make_chunk("new", [0.0, 1.0])])
    assert len(retriever) == 1
    assert retriever._chunks[0].id == "existing"
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_missing_faiss_fails_only_when_backend_is_constructed() -> None:
    script = """
import builtins
real_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'faiss':
        raise ImportError('blocked for test')
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked
from ragframework.retriever import InMemoryRetriever, FAISSRetriever
InMemoryRetriever()
try:
    FAISSRetriever()
except ImportError as exc:
    assert 'pip install ragframework[faiss]' in str(exc)
else:
    raise AssertionError('FAISSRetriever did not reject a missing dependency')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
