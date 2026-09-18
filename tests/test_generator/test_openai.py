"""Tests for the OpenAI generator."""

import builtins
import sys
import types

import pytest

from ragframework.base import Chunk
from ragframework.exceptions import GeneratorError
from ragframework.generator.openai import OpenAIGenerator


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    generator = OpenAIGenerator()

    with pytest.raises(GeneratorError, match="API key is required"):
        generator.generate("What is RAG?", [])


def test_import_error_is_helpful(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "openai":
            raise ImportError("No module named 'openai'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    generator = OpenAIGenerator()
    with pytest.raises(GeneratorError, match=r"ragframework\[openai\]"):
        generator.generate("What is RAG?", [])


def test_generates_grounded_answer(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = {}

    class FakeMessage:
        content = "RAG retrieves relevant context before generation."

    class FakeResponse:
        choices = [types.SimpleNamespace(message=FakeMessage())]

    class FakeCompletions:
        def create(self, **kwargs):
            calls.update(kwargs)
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key):
            calls["api_key"] = api_key
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeClient))

    generator = OpenAIGenerator(model="test-model", max_tokens=128)
    context = [Chunk(id="1", content="RAG retrieves context.")]

    answer = generator.generate("What is RAG?", context)

    assert answer == "RAG retrieves relevant context before generation."
    assert calls["api_key"] == "test-key"
    assert calls["model"] == "test-model"
    assert calls["max_tokens"] == 128
    assert calls["messages"][0]["role"] == "system"
    assert "RAG retrieves context." in calls["messages"][1]["content"]
    assert "What is RAG?" in calls["messages"][1]["content"]


def test_wraps_api_errors(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeCompletions:
        def create(self, **kwargs):
            raise RuntimeError("service unavailable")

    class FakeClient:
        def __init__(self, api_key):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=FakeClient))

    generator = OpenAIGenerator()
    with pytest.raises(GeneratorError, match="OpenAI generation failed"):
        generator.generate("What is RAG?", [])
