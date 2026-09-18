"""Tests for the Anthropic Messages API generator."""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from ragframework.base import Chunk, Generator
from ragframework.exceptions import GeneratorError
from ragframework.generator.anthropic import AnthropicGenerator


class FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class FakeNonTextBlock:
    def __init__(self) -> None:
        self.type = "tool_use"


class FakeMessageResponse:
    def __init__(self, content: list[object]) -> None:
        self.content = content


class FakeMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.create_side_effect: Exception | None = None
        self.response_content: list[object] = [FakeTextBlock("Generated answer.")]

    def create(self, **kwargs: object) -> FakeMessageResponse:
        self.calls.append(kwargs)
        if self.create_side_effect is not None:
            raise self.create_side_effect
        return FakeMessageResponse(self.response_content)


class FakeAnthropicClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.messages = FakeMessages()


@pytest.fixture
def fake_anthropic(monkeypatch: pytest.MonkeyPatch) -> list[FakeAnthropicClient]:
    module = types.ModuleType("anthropic")
    created_clients: list[FakeAnthropicClient] = []

    def factory(*, api_key: str | None = None) -> FakeAnthropicClient:
        client = FakeAnthropicClient(api_key=api_key)
        created_clients.append(client)
        return client

    module.Anthropic = factory
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return created_clients


def _make_generator(
    fake_anthropic: list[FakeAnthropicClient],
    *,
    model: str = "claude-sonnet-5",
    api_key: str = "sk-test",
    **kwargs: object,
) -> tuple[AnthropicGenerator, FakeAnthropicClient]:
    generator = AnthropicGenerator(model=model, api_key=api_key, **kwargs)
    return generator, fake_anthropic[-1]


def test_subclasses_generator() -> None:
    assert issubclass(AnthropicGenerator, Generator)


def test_successful_generation(fake_anthropic: list[FakeAnthropicClient]) -> None:
    generator, client = _make_generator(fake_anthropic)
    client.messages.response_content = [FakeTextBlock("Grounded answer.")]

    answer = generator.generate(
        "What is RAG?", [Chunk(id="c1", content="RAG combines retrieval and generation.")]
    )

    assert answer == "Grounded answer."
    assert len(client.messages.calls) == 1


def test_api_call_arguments(fake_anthropic: list[FakeAnthropicClient]) -> None:
    generator, client = _make_generator(
        fake_anthropic,
        model="claude-test-model",
        system_prompt="Stay grounded.",
        max_tokens=256,
    )

    generator.generate("Explain embeddings.", [Chunk(id="c1", content="Vectors represent text.")])

    call = client.messages.calls[0]
    assert call["model"] == "claude-test-model"
    assert call["max_tokens"] == 256
    assert call["system"] == "Stay grounded."
    messages = call["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], str)


def test_context_formatting(fake_anthropic: list[FakeAnthropicClient]) -> None:
    generator, client = _make_generator(fake_anthropic)
    context = [
        Chunk(id="alpha-id", content="Alpha", metadata={"source": "a.txt"}, embedding=[1.0]),
        Chunk(id="beta-id", content="Beta", metadata={"source": "b.txt"}, embedding=[2.0]),
    ]

    generator.generate("What happened?", context)

    user_message = client.messages.calls[0]["messages"][0]["content"]
    assert "[1] Alpha" in user_message
    assert "[2] Beta" in user_message
    assert user_message.index("[1] Alpha") < user_message.index("[2] Beta")
    assert "Question: What happened?" in user_message
    assert "alpha-id" not in user_message
    assert "a.txt" not in user_message


def test_empty_context_still_calls_api(fake_anthropic: list[FakeAnthropicClient]) -> None:
    generator, client = _make_generator(fake_anthropic)

    generator.generate("q", [])

    assert len(client.messages.calls) == 1
    user_message = client.messages.calls[0]["messages"][0]["content"]
    assert "Question: q" in user_message
    assert "[1]" not in user_message


def test_explicit_api_key(fake_anthropic: list[FakeAnthropicClient]) -> None:
    generator = AnthropicGenerator(model="claude-sonnet-5", api_key="sk-test")

    assert fake_anthropic[-1].api_key == "sk-test"
    assert generator.model == "claude-sonnet-5"


def test_environment_api_key_fallback(
    fake_anthropic: list[FakeAnthropicClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")

    AnthropicGenerator(model="claude-sonnet-5")

    assert fake_anthropic[-1].api_key == "env-key"


def test_missing_api_key(
    fake_anthropic: list[FakeAnthropicClient], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(GeneratorError, match="No API key provided"):
        AnthropicGenerator(model="claude-sonnet-5")

    assert fake_anthropic == []


def test_missing_dependency_has_helpful_message(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "anthropic":
            raise ImportError("No module named 'anthropic'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "anthropic", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"ragframework\[anthropic\]"):
        AnthropicGenerator(model="claude-sonnet-5", api_key="sk-test")


def test_client_construction_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("anthropic")

    def failing_factory(*, api_key: str | None = None) -> None:
        raise RuntimeError("client init failed")

    module.Anthropic = failing_factory
    monkeypatch.setitem(sys.modules, "anthropic", module)

    with pytest.raises(GeneratorError, match="Could not create Anthropic client") as exc_info:
        AnthropicGenerator(model="claude-sonnet-5", api_key="sk-test")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_api_failure(fake_anthropic: list[FakeAnthropicClient]) -> None:
    generator, client = _make_generator(fake_anthropic)
    client.messages.create_side_effect = RuntimeError("API offline")

    with pytest.raises(GeneratorError, match="Anthropic API call failed") as exc_info:
        generator.generate("q", [Chunk(id="c1", content="context")])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_multiple_text_blocks(fake_anthropic: list[FakeAnthropicClient]) -> None:
    generator, client = _make_generator(fake_anthropic)
    client.messages.response_content = [FakeTextBlock("Part one. "), FakeTextBlock("Part two.")]

    answer = generator.generate("q", [Chunk(id="c1", content="context")])

    assert answer == "Part one. Part two."


def test_empty_response_raises_generator_error(fake_anthropic: list[FakeAnthropicClient]) -> None:
    generator, client = _make_generator(fake_anthropic)
    client.messages.response_content = []

    with pytest.raises(GeneratorError, match="no content blocks"):
        generator.generate("q", [Chunk(id="c1", content="context")])


def test_non_text_response_raises_generator_error(
    fake_anthropic: list[FakeAnthropicClient],
) -> None:
    generator, client = _make_generator(fake_anthropic)
    client.messages.response_content = [FakeNonTextBlock()]

    with pytest.raises(GeneratorError, match="no text blocks"):
        generator.generate("q", [Chunk(id="c1", content="context")])


def test_constructor_configuration(fake_anthropic: list[FakeAnthropicClient]) -> None:
    generator = AnthropicGenerator(
        model="custom-model",
        api_key="sk-test",
        system_prompt="Custom prompt.",
        max_tokens=512,
    )

    assert generator.model == "custom-model"
    assert generator.system_prompt == "Custom prompt."
    assert generator.max_tokens == 512
