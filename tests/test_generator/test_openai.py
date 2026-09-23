"""Tests for the OpenAI generator."""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from ragframework.base import Chunk, Generator
from ragframework.exceptions import GeneratorError
from ragframework.generator.openai import OpenAIGenerator


class FakeOpenAIResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.create_side_effect: Exception | None = None
        self.response_content: str | None = "Generated answer."

    def create(self, **kwargs: object) -> FakeOpenAIResponse:
        self.calls.append(kwargs)
        if self.create_side_effect is not None:
            raise self.create_side_effect
        return FakeOpenAIResponse(self.response_content)


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeOpenAIClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self.chat = FakeChat()


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> list[FakeOpenAIClient]:
    module = types.ModuleType("openai")
    created_clients: list[FakeOpenAIClient] = []

    def factory(*, api_key: str | None = None) -> FakeOpenAIClient:
        client = FakeOpenAIClient(api_key=api_key)
        created_clients.append(client)
        return client

    module.OpenAI = factory
    monkeypatch.setitem(sys.modules, "openai", module)
    return created_clients


def _make_generator(
    fake_openai: list[FakeOpenAIClient],
    *,
    model: str = "gpt-4o-mini",
    api_key: str = "test-key",
    **kwargs: object,
) -> tuple[OpenAIGenerator, FakeOpenAIClient]:
    generator = OpenAIGenerator(model=model, api_key=api_key, **kwargs)
    return generator, fake_openai[-1]


def test_subclasses_generator() -> None:
    assert issubclass(OpenAIGenerator, Generator)


def test_successful_generation(fake_openai: list[FakeOpenAIClient]) -> None:
    generator, client = _make_generator(fake_openai)
    client.chat.completions.response_content = "The answer is 42."

    answer = generator.generate(
        "What is the answer?",
        [Chunk(id="1", content="The answer is 42.")],
    )

    assert answer == "The answer is 42."
    assert len(client.chat.completions.calls) == 1


def test_api_call_arguments(fake_openai: list[FakeOpenAIClient]) -> None:
    generator, client = _make_generator(
        fake_openai,
        model="custom-model",
        system_prompt="Use only the supplied context.",
        max_tokens=100,
    )

    generator.generate("Question?", [])

    call = client.chat.completions.calls[0]

    assert call["model"] == "custom-model"
    assert call["max_tokens"] == 100

    messages = call["messages"]
    assert isinstance(messages, list)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Use only the supplied context."
    assert messages[1]["role"] == "user"


def test_context_formatting(fake_openai: list[FakeOpenAIClient]) -> None:
    generator, client = _make_generator(fake_openai)

    context = [
        Chunk(id="alpha-id", content="Alpha"),
        Chunk(id="beta-id", content="Beta"),
    ]

    generator.generate("What happened?", context)

    messages = client.chat.completions.calls[0]["messages"]
    user_message = messages[1]["content"]

    assert "[Chunk 1]" in user_message
    assert "Alpha" in user_message
    assert "[Chunk 2]" in user_message
    assert "Beta" in user_message
    assert user_message.index("[Chunk 1]") < user_message.index("[Chunk 2]")
    assert "Question:\nWhat happened?" in user_message


def test_empty_context(fake_openai: list[FakeOpenAIClient]) -> None:
    generator, client = _make_generator(fake_openai)

    generator.generate("Question?", [])

    messages = client.chat.completions.calls[0]["messages"]
    user_message = messages[1]["content"]

    assert "No context was retrieved." in user_message
    assert "Question:\nQuestion?" in user_message


def test_explicit_api_key(fake_openai: list[FakeOpenAIClient]) -> None:
    generator = OpenAIGenerator(api_key="sk-test")

    assert fake_openai[-1].api_key == "sk-test"
    assert generator.model == "gpt-4o-mini"


def test_environment_api_key_fallback(
    fake_openai: list[FakeOpenAIClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    OpenAIGenerator()

    assert fake_openai[-1].api_key == "env-key"


def test_missing_api_key(
    fake_openai: list[FakeOpenAIClient],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(GeneratorError, match="No API key provided"):
        OpenAIGenerator()

    assert fake_openai == []


def test_missing_dependency_has_helpful_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "openai", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match=r"ragframework\[openai\]"):
        OpenAIGenerator(api_key="sk-test")


def test_client_construction_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("openai")

    def failing_factory(*, api_key: str | None = None) -> None:
        raise RuntimeError("client init failed")

    module.OpenAI = failing_factory
    monkeypatch.setitem(sys.modules, "openai", module)

    with pytest.raises(GeneratorError, match="Could not initialize OpenAI client") as exc_info:
        OpenAIGenerator(api_key="sk-test")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_api_failure(fake_openai: list[FakeOpenAIClient]) -> None:
    generator, client = _make_generator(fake_openai)
    client.chat.completions.create_side_effect = RuntimeError("API unavailable")

    with pytest.raises(GeneratorError, match="OpenAI generation failed") as exc_info:
        generator.generate("Question?", [Chunk(id="1", content="context")])

    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_empty_response_raises_generator_error(
    fake_openai: list[FakeOpenAIClient],
) -> None:
    generator, client = _make_generator(fake_openai)
    client.chat.completions.response_content = ""

    with pytest.raises(GeneratorError, match="empty response"):
        generator.generate("Question?", [])


def test_constructor_configuration(fake_openai: list[FakeOpenAIClient]) -> None:
    generator = OpenAIGenerator(
        model="custom-model",
        api_key="sk-test",
        system_prompt="Custom prompt.",
        max_tokens=512,
    )

    assert generator.model == "custom-model"
    assert generator.system_prompt == "Custom prompt."
    assert generator.max_tokens == 512
