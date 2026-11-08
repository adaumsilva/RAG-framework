"""Tests for BM25Retriever."""

import pytest

from ragframework.base import Chunk, Retriever
from ragframework.exceptions import RetrieverError
from ragframework.retriever.bm25 import BM25Retriever


def test_subclasses_retriever() -> None:
    assert issubclass(BM25Retriever, Retriever)


def test_default_variant_is_bm25l() -> None:
    assert BM25Retriever().variant == "l"


def test_add_and_retrieve_lexical() -> None:
    retriever = BM25Retriever()
    chunk1 = Chunk(id="1", content="Python is a popular programming language")
    chunk2 = Chunk(id="2", content="Rust provides memory safety without garbage collection")
    chunk3 = Chunk(id="3", content="Cats are small carnivorous mammals")
    retriever.add([chunk1, chunk2, chunk3])

    results = retriever.retrieve(query_embedding=[0.0], query_text="programming in Python", top_k=2)
    assert len(results) >= 1
    assert results[0].id == "1"


def test_retrieve_with_small_corpus() -> None:
    # The default tokenizer is intentionally lowercase + whitespace split.
    retriever = BM25Retriever()
    chunk1 = Chunk(id="1", content="hi, my name is Alice I am an engineer.")
    chunk2 = Chunk(id="2", content="hi, my name is Bob I am a scientist.")
    retriever.add([chunk1, chunk2])

    results = retriever.retrieve(query_embedding=[0.0], query_text="Alice", top_k=1)
    assert len(results) == 1
    assert results[0].id == "1"


def test_keyword_query_string() -> None:
    retriever = BM25Retriever()
    chunk1 = Chunk(id="1", content="machine learning with neural networks")
    chunk2 = Chunk(id="2", content="database optimization with indexes")
    retriever.add([chunk1, chunk2])

    results = retriever.retrieve(query_embedding=[0.0], query_text="neural networks", top_k=1)
    assert len(results) == 1
    assert results[0].id == "1"


def test_empty_retriever_returns_empty() -> None:
    retriever = BM25Retriever()
    assert retriever.retrieve(query_embedding=[0.0], query_text="anything") == []
    assert len(retriever) == 0


def test_unmatched_query_returns_empty_for_non_empty_corpus() -> None:
    retriever = BM25Retriever()
    retriever.add([Chunk(id="1", content="apple"), Chunk(id="2", content="banana")])

    assert retriever.retrieve(query_embedding=[0.0], query_text="unknown") == []


@pytest.mark.parametrize("variant", ["okapi", "l", "plus", "OKAPI"])
def test_supported_variants_retrieve(variant: str) -> None:
    retriever = BM25Retriever(variant=variant)
    retriever.add([Chunk(id="match", content="error code ERR-42")])

    results = retriever.retrieve(query_embedding=[0.0], query_text="ERR-42")

    assert results[0].id == "match"


def test_add_after_first_query_rebuilds_index() -> None:
    retriever = BM25Retriever()
    retriever.add([Chunk(id="first", content="first document")])
    assert retriever.retrieve(query_embedding=[0.0], query_text="second") == []

    second = Chunk(id="second", content="second document")
    retriever.add([second])

    assert retriever.retrieve(query_embedding=[0.0], query_text="second") == [second]


def test_empty_add_is_a_noop() -> None:
    retriever = BM25Retriever()
    retriever.add([])

    assert len(retriever) == 0
    assert retriever.retrieve(query_embedding=[0.0], query_text="anything") == []


def test_invalid_chunk_does_not_partially_modify_corpus() -> None:
    retriever = BM25Retriever()
    valid = Chunk(id="valid", content="valid")
    invalid = Chunk(id="invalid", content=None)  # type: ignore[arg-type]

    with pytest.raises(RetrieverError, match="has no text content"):
        retriever.add([valid, invalid])

    assert len(retriever) == 0


def test_len() -> None:
    retriever = BM25Retriever()
    chunks = [Chunk(id=str(i), content=f"Document number {i}") for i in range(5)]
    retriever.add(chunks)
    assert len(retriever) == 5


def test_chunks_do_not_require_embeddings() -> None:
    retriever = BM25Retriever()
    chunk = Chunk(id="no_emb", content="Content with None embedding", embedding=None)
    retriever.add([chunk])
    assert len(retriever) == 1
    results = retriever.retrieve(query_embedding=[0.0], query_text="None embedding")
    assert len(results) == 1
    assert results[0].id == "no_emb"


def test_missing_query_text_raises() -> None:
    retriever = BM25Retriever()
    retriever.add([Chunk(id="1", content="test doc")])
    with pytest.raises(RetrieverError, match="requires query_text"):
        retriever.retrieve(query_embedding=[0.1, 0.2])


def test_non_string_query_text_raises() -> None:
    retriever = BM25Retriever()

    with pytest.raises(RetrieverError, match="query_text must be a string"):
        retriever.retrieve(query_embedding=[0.0], query_text=123)  # type: ignore[arg-type]


def test_invalid_top_k_raises() -> None:
    retriever = BM25Retriever()
    retriever.add([Chunk(id="1", content="test doc")])
    with pytest.raises(RetrieverError, match="top_k must be an integer"):
        retriever.retrieve(query_embedding=[0.0], query_text="test", top_k="invalid")  # type: ignore[arg-type]
    with pytest.raises(RetrieverError, match="top_k must be an integer"):
        retriever.retrieve(query_embedding=[0.0], query_text="test", top_k=True)  # type: ignore[arg-type]


def test_non_positive_top_k_returns_empty() -> None:
    retriever = BM25Retriever()
    retriever.add([Chunk(id="1", content="test doc")])
    assert retriever.retrieve(query_embedding=[0.0], query_text="test", top_k=0) == []
    assert retriever.retrieve(query_embedding=[0.0], query_text="test", top_k=-1) == []


def test_invalid_variant_raises() -> None:
    with pytest.raises(RetrieverError, match="Invalid BM25 variant"):
        BM25Retriever(variant="nonexistent")


@pytest.mark.parametrize("k1", [-1, 0, True, float("nan"), float("inf")])
def test_invalid_k1_raises(k1: object) -> None:
    with pytest.raises(RetrieverError, match="k1"):
        BM25Retriever(k1=k1)  # type: ignore[arg-type]


@pytest.mark.parametrize("b", [-0.1, 1.1, True, float("nan"), float("inf")])
def test_invalid_b_raises(b: object) -> None:
    with pytest.raises(RetrieverError, match="b"):
        BM25Retriever(b=b)  # type: ignore[arg-type]


def test_non_callable_tokenizer_raises() -> None:
    with pytest.raises(RetrieverError, match="tokenizer must be callable"):
        BM25Retriever(tokenizer="not-callable")  # type: ignore[arg-type]


def test_custom_tokenizer() -> None:
    def simple_tokens(text: str) -> list[str]:
        return text.split()

    retriever = BM25Retriever(tokenizer=simple_tokens)
    retriever.add([Chunk(id="1", content="hello world")])
    results = retriever.retrieve(query_embedding=[0.0], query_text="hello")
    assert len(results) == 1
    assert results[0].id == "1"


def test_okapi_keeps_matches_with_zero_scores() -> None:
    retriever = BM25Retriever(variant="okapi")
    retriever.add(
        [
            Chunk(id="incident", content="status"),
            Chunk(id="other", content="status"),
        ]
    )

    results = retriever.retrieve(query_embedding=[0.0], query_text="status")

    assert len(results) == 2
    assert {chunk.id for chunk in results} == {"incident", "other"}
