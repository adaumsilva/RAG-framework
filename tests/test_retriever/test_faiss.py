"""Tests for the FAISS-backed retriever."""

import json
import logging
import subprocess
import sys

import faiss
import numpy as np
import pytest

import ragframework.retriever.faiss as faiss_retriever_module
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


def test_add_logs_count_and_dimension(caplog) -> None:
    retriever = FAISSRetriever()
    chunks = [
        make_chunk("first", [1.0, 0.0, 0.0]),
        make_chunk("second", [0.0, 1.0, 0.0]),
    ]

    with caplog.at_level(
        logging.DEBUG,
        logger="ragframework.retriever.faiss",
    ):
        retriever.add(chunks)

    assert "Added chunks count=2 dimension=3" in caplog.text


def test_clear_empties_index_and_allows_new_dimension() -> None:
    retriever = FAISSRetriever()
    retriever.add([make_chunk("old", [1.0, 0.0])])
    retriever.clear()
    assert len(retriever) == 0
    assert retriever.retrieve([1.0, 0.0]) == []
    new = make_chunk("new", [1.0, 0.0, 0.0])
    retriever.add([new])
    assert retriever.retrieve([1.0, 0.0, 0.0]) == [new]


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


def test_save_and_load_round_trip(tmp_path) -> None:
    original = FAISSRetriever(m=12, ef_construction=48, ef_search=64)
    original.add(
        [
            make_chunk("north", [10.0, 0.0]),
            make_chunk("northeast", [1.0, 1.0]),
            make_chunk("east", [0.0, 3.0]),
        ]
    )

    original.save(tmp_path)
    loaded = FAISSRetriever.load(tmp_path)

    assert len(loaded) == len(original)
    assert loaded._m == 12
    assert loaded._ef_construction == 48
    assert loaded._ef_search == 64
    query = [2.0, 0.5]
    original_ids = [c.id for c in original.retrieve(query, top_k=3)]
    loaded_ids = [c.id for c in loaded.retrieve(query, top_k=3)]
    assert loaded_ids == original_ids
    assert loaded.retrieve(query, top_k=1)[0].metadata == {"source": "north"}


def test_load_missing_directory_raises(tmp_path) -> None:
    with pytest.raises(RetrieverError, match="Could not load"):
        FAISSRetriever.load(tmp_path / "nope")


def test_load_corrupt_index_raises(tmp_path) -> None:
    retriever = FAISSRetriever()
    retriever.add([make_chunk("one", [1.0, 0.0])])
    retriever.save(tmp_path)
    (tmp_path / "index.faiss").write_bytes(b"not a faiss index")

    with pytest.raises(RetrieverError, match="Could not load"):
        FAISSRetriever.load(tmp_path)


def test_load_wrong_format_version_raises(tmp_path) -> None:
    retriever = FAISSRetriever()
    retriever.add([make_chunk("one", [1.0, 0.0])])
    retriever.save(tmp_path)

    meta_file = tmp_path / "meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["format_version"] = 999
    meta_file.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(RetrieverError, match="format"):
        FAISSRetriever.load(tmp_path)


def test_load_without_faiss_raises_import_error(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = FAISSRetriever()
    retriever.add([make_chunk("one", [1.0, 0.0])])
    retriever.save(tmp_path)

    # Simulate the optional dependency being absent, as when the import fails.
    monkeypatch.setattr(faiss_retriever_module, "_FAISS_AVAILABLE", False)
    monkeypatch.delattr(faiss_retriever_module, "faiss")

    with pytest.raises(ImportError, match=r"pip install ragframework\[faiss\]"):
        FAISSRetriever.load(tmp_path)


def _save_two_chunks(directory) -> tuple[dict, list]:
    retriever = FAISSRetriever()
    retriever.add([make_chunk("a", [1.0, 0.0]), make_chunk("b", [0.0, 1.0])])
    retriever.save(directory)
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    records = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
    return meta, records


def test_load_dimension_mismatch_raises(tmp_path) -> None:
    meta, _records = _save_two_chunks(tmp_path)
    meta["dimension"] = 3
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(RetrieverError, match="dimension 2 but meta.json says 3"):
        FAISSRetriever.load(tmp_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda meta: [], "meta.json.*expected a JSON object"),
        (lambda meta: {k: v for k, v in meta.items() if k != "m"}, "'m' must be a positive"),
        (lambda meta: {**meta, "m": "32"}, "'m' must be a positive"),
        (lambda meta: {**meta, "ef_search": 0}, "'ef_search' must be a positive"),
        (lambda meta: {**meta, "dimension": True}, "'dimension' must be a positive"),
    ],
)
def test_load_malformed_meta_json_raises(tmp_path, mutate, message: str) -> None:
    meta, _records = _save_two_chunks(tmp_path)
    (tmp_path / "meta.json").write_text(json.dumps(mutate(meta)), encoding="utf-8")

    with pytest.raises(RetrieverError, match=message):
        FAISSRetriever.load(tmp_path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda records: {"chunks": records}, "chunks.json.*list of chunk records"),
        (lambda records: [{"id": 1, "content": "x", "metadata": {}}] * 2, "needs 'id'"),
        (lambda records: records[:1], "1 chunks but 2 vectors"),
    ],
)
def test_load_malformed_chunks_json_raises(tmp_path, mutate, message: str) -> None:
    _meta, records = _save_two_chunks(tmp_path)
    (tmp_path / "chunks.json").write_text(json.dumps(mutate(records)), encoding="utf-8")

    with pytest.raises(RetrieverError, match=message):
        FAISSRetriever.load(tmp_path)


def test_load_rejects_index_with_wrong_metric(tmp_path) -> None:
    _save_two_chunks(tmp_path)
    l2_index = faiss.IndexHNSWFlat(2, 32)
    l2_index.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    faiss.write_index(l2_index, str(tmp_path / "index.faiss"))

    with pytest.raises(RetrieverError, match="inner-product metric"):
        FAISSRetriever.load(tmp_path)
