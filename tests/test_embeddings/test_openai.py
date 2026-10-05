"""Tests for the OpenAI embeddings integration, with no external API calls."""

from __future__ import annotations

import asyncio
import builtins
import sys
import types
from unittest.mock import AsyncMock, Mock

import pytest

from ragframework.base import Embedder
from ragframework.embeddings.openai import OpenAIEmbedder
from ragframework.exceptions import EmbedderError


class FakeOpenAIError(Exception):
    pass


class FakeAuthenticationError(FakeOpenAIError):
    pass


class FakeRateLimitError(FakeOpenAIError):
    pass


class FakeBadRequestError(FakeOpenAIError):
    pass


class FakeEncoder:
    """Use character count as a predictable stand-in for token count."""

    def encode_ordinary_batch(self, texts: list[str]) -> list[list[int]]:
        return [list(range(len(text))) for text in texts]

    @staticmethod
    def decode(tokens: list[int]) -> str:
        return "x" * len(tokens)


def _response(input_texts: list[str], *, omit_last: bool = False) -> types.SimpleNamespace:
    data = [
        types.SimpleNamespace(index=index, embedding=[float(index), float(index + 10)])
        for index in range(len(input_texts) - int(omit_last))
    ]
    return types.SimpleNamespace(data=list(reversed(data)))


async def _async_create(**kwargs: object) -> types.SimpleNamespace:
    return _response(kwargs["input"])


@pytest.fixture
def mocked_dependencies(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Install fake optional packages and clients into this test's import table."""
    encoder = FakeEncoder()
    tokenizer = types.ModuleType("tiktoken")
    tokenizer.encoding_for_model = Mock(return_value=encoder)
    tokenizer.get_encoding = Mock(return_value=encoder)

    sync_client = types.SimpleNamespace(embeddings=types.SimpleNamespace(create=Mock()))
    async_client = types.SimpleNamespace(embeddings=types.SimpleNamespace(create=AsyncMock()))
    openai = types.ModuleType("openai")
    openai.OpenAIError = FakeOpenAIError
    openai.AuthenticationError = FakeAuthenticationError
    openai.RateLimitError = FakeRateLimitError
    openai.BadRequestError = FakeBadRequestError
    openai.OpenAI = Mock(return_value=sync_client)
    openai.AsyncOpenAI = Mock(return_value=async_client)

    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setitem(sys.modules, "tiktoken", tokenizer)
    return {
        "openai": openai,
        "tokenizer": tokenizer,
        "encoder": encoder,
        "sync_client": sync_client,
        "async_client": async_client,
    }


def test_subclasses_embedder_and_uses_default_model(
    mocked_dependencies: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    embedder = OpenAIEmbedder()

    assert isinstance(embedder, Embedder)
    assert embedder.model == "text-embedding-3-small"
    assert embedder._api_key == "env-key"


def test_openai_embedder_subclasses_embedder() -> None:
    assert issubclass(OpenAIEmbedder, Embedder)


def test_explicit_model_and_api_key_override_environment(
    mocked_dependencies: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    embedder = OpenAIEmbedder(model="custom-model", api_key="explicit-key")

    assert embedder.model == "custom-model"
    assert embedder._api_key == "explicit-key"


def test_missing_api_key_raises_embedder_error(
    mocked_dependencies: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(EmbedderError, match="API key not found"):
        OpenAIEmbedder()


def test_missing_openai_dependency_has_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "openai", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"ragframework\[openai\]"):
        OpenAIEmbedder(api_key="test-key")


def test_missing_tiktoken_dependency_has_install_hint(
    mocked_dependencies: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tiktoken":
            raise ImportError("No module named 'tiktoken'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "tiktoken", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    embedder = OpenAIEmbedder(api_key="test-key")

    with pytest.raises(ImportError, match=r"ragframework\[tokens\]"):
        embedder.embed(["text"])


def test_client_initialization_failure_is_wrapped(mocked_dependencies: dict[str, object]) -> None:
    mocked_dependencies["openai"].OpenAI.side_effect = RuntimeError("client setup failed")
    embedder = OpenAIEmbedder(api_key="test-key")

    with pytest.raises(EmbedderError, match="Could not initialize OpenAI client") as exc_info:
        embedder.embed(["text"])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_tokenizer_failure_is_wrapped(mocked_dependencies: dict[str, object]) -> None:
    encoder = mocked_dependencies["encoder"]

    def fail_encoding(texts: list[str]) -> list[list[int]]:
        raise RuntimeError("tokenizer failed")

    encoder.encode_ordinary_batch = fail_encoding
    embedder = OpenAIEmbedder(api_key="test-key")

    with pytest.raises(EmbedderError, match="Could not prepare embedding batches") as exc_info:
        embedder.embed(["text"])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_sync_embedding_preserves_order_and_sends_model_and_key(
    mocked_dependencies: dict[str, object],
) -> None:
    client = mocked_dependencies["sync_client"]
    create = client.embeddings.create
    create.side_effect = lambda **kwargs: _response(kwargs["input"])
    embedder = OpenAIEmbedder(model="custom-model", api_key="test-key")

    result = embedder.embed(["first", "second"])

    assert result == [[0.0, 10.0], [1.0, 11.0]]
    assert create.call_args.kwargs == {"model": "custom-model", "input": ["first", "second"]}
    mocked_dependencies["openai"].OpenAI.assert_called_once_with(
        api_key="test-key",
        organization=None,
        base_url=None,
        timeout=30.0,
        max_retries=5,
    )


def test_sync_requests_respect_item_and_token_batch_limits(
    mocked_dependencies: dict[str, object],
) -> None:
    client = mocked_dependencies["sync_client"]
    create = client.embeddings.create
    create.side_effect = lambda **kwargs: _response(kwargs["input"])
    embedder = OpenAIEmbedder(
        api_key="test-key",
        batch_size=2,
        max_batch_tokens=5,
        max_input_tokens=5,
    )

    result = embedder.embed(["aa", "bbb", "c", "dd"])

    assert len(result) == 4
    assert [call.kwargs["input"] for call in create.call_args_list] == [
        ["aa", "bbb"],
        ["c", "dd"],
    ]


def test_too_long_text_fails_before_creating_client(mocked_dependencies: dict[str, object]) -> None:
    embedder = OpenAIEmbedder(api_key="test-key", max_input_tokens=2)

    with pytest.raises(EmbedderError, match="exceed 2 tokens"):
        embedder.embed(["long"])

    mocked_dependencies["openai"].OpenAI.assert_not_called()


def test_truncate_overflow_sends_truncated_text(mocked_dependencies: dict[str, object]) -> None:
    client = mocked_dependencies["sync_client"]
    create = client.embeddings.create
    create.side_effect = lambda **kwargs: _response(kwargs["input"])
    embedder = OpenAIEmbedder(api_key="test-key", max_input_tokens=2, on_overflow="truncate")

    embedder.embed(["long"])

    assert create.call_args.kwargs["input"] == ["xx"]


def test_wrong_response_count_stops_before_later_sync_batches(
    mocked_dependencies: dict[str, object],
) -> None:
    client = mocked_dependencies["sync_client"]
    create = client.embeddings.create
    create.side_effect = lambda **kwargs: _response(kwargs["input"], omit_last=True)
    embedder = OpenAIEmbedder(api_key="test-key", batch_size=1)

    with pytest.raises(EmbedderError, match="Expected 1 embeddings for batch, got 0"):
        embedder.embed(["first", "second"])

    assert create.call_count == 1


def test_openai_api_failure_is_wrapped_and_chained(mocked_dependencies: dict[str, object]) -> None:
    client = mocked_dependencies["sync_client"]
    client.embeddings.create.side_effect = FakeRateLimitError("rate limited")
    embedder = OpenAIEmbedder(api_key="test-key")

    with pytest.raises(EmbedderError, match="Rate limited") as exc_info:
        embedder.embed(["text"])

    assert isinstance(exc_info.value.__cause__, FakeRateLimitError)


def test_non_sdk_request_failure_is_wrapped(mocked_dependencies: dict[str, object]) -> None:
    client = mocked_dependencies["sync_client"]
    client.embeddings.create.side_effect = RuntimeError("transport failed")
    embedder = OpenAIEmbedder(api_key="test-key")

    with pytest.raises(EmbedderError, match="Could not create embeddings") as exc_info:
        embedder.embed(["text"])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_empty_and_invalid_inputs(mocked_dependencies: dict[str, object]) -> None:
    embedder = OpenAIEmbedder(api_key="test-key")

    assert embedder.embed([]) == []
    with pytest.raises(TypeError, match="Expected a list of strings"):
        embedder.embed(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Expected a list of strings"):
        embedder.embed(("text",))  # type: ignore[arg-type]
    with pytest.raises(EmbedderError, match="Empty/whitespace-only"):
        embedder.embed(["  "])


@pytest.mark.asyncio
async def test_async_embedding_preserves_order_and_respects_batching(
    mocked_dependencies: dict[str, object],
) -> None:
    client = mocked_dependencies["async_client"]
    create = client.embeddings.create
    create.side_effect = _async_create
    embedder = OpenAIEmbedder(api_key="test-key", batch_size=2)

    result = await embedder.aembed(["first", "second", "third"])

    assert result == [[0.0, 10.0], [1.0, 11.0], [0.0, 10.0]]
    assert [call.kwargs["input"] for call in create.call_args_list] == [
        ["first", "second"],
        ["third"],
    ]


@pytest.mark.asyncio
async def test_async_failure_is_wrapped_and_empty_list_is_valid(
    mocked_dependencies: dict[str, object],
) -> None:
    client = mocked_dependencies["async_client"]
    client.embeddings.create.side_effect = FakeAuthenticationError("bad key")
    embedder = OpenAIEmbedder(api_key="test-key")

    assert await embedder.aembed([]) == []
    with pytest.raises(EmbedderError, match="Authentication failed") as exc_info:
        await embedder.aembed(["text"])

    assert isinstance(exc_info.value.__cause__, FakeAuthenticationError)


@pytest.mark.asyncio
async def test_async_invalid_input_and_cancellation_propagate(
    mocked_dependencies: dict[str, object],
) -> None:
    embedder = OpenAIEmbedder(api_key="test-key")
    with pytest.raises(TypeError, match="Expected a list of strings"):
        await embedder.aembed(None)  # type: ignore[arg-type]

    client = mocked_dependencies["async_client"]
    client.embeddings.create.side_effect = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await embedder.aembed(["text"])
