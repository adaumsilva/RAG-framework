import os
from unittest.mock import MagicMock, patch

import pytest

from ragframework.exceptions import EmbedderError


def test_openai_embedder_missing_package():
    """Test that EmbedderError is raised if openai is not installed."""
    with patch("ragframework.embeddings.openai.openai", None):
        from ragframework.embeddings.openai import OpenAIEmbedder

        with pytest.raises(EmbedderError, match="The `openai` package is not installed"):
            OpenAIEmbedder(api_key="fake-key")


@patch("ragframework.embeddings.openai.openai", MagicMock())
def test_openai_embedder_missing_api_key():
    """Test that EmbedderError is raised if API key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        from ragframework.embeddings.openai import OpenAIEmbedder

        with pytest.raises(EmbedderError, match="No API key provided"):
            OpenAIEmbedder()


@patch("ragframework.embeddings.openai.openai", MagicMock())
def test_openai_embedder_instantiation():
    """Test successful instantiation with an API key."""
    from ragframework.embeddings.openai import OpenAIEmbedder

    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}, clear=True):
        embedder = OpenAIEmbedder(model="custom-model", batch_size=100)
        assert embedder.model == "custom-model"
        assert embedder.batch_size == 100
        import ragframework.embeddings.openai

        ragframework.embeddings.openai.openai.OpenAI.assert_called_with(api_key="env-key")


@patch("ragframework.embeddings.openai.openai", MagicMock())
def test_openai_embedder_empty_texts():
    """Test that empty texts return empty results without API calls."""
    from ragframework.embeddings.openai import OpenAIEmbedder

    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
        embedder = OpenAIEmbedder()
        assert embedder.embed([]) == []


@patch("ragframework.embeddings.openai.openai")
def test_openai_embedder_embed_success(mock_openai):
    """Test successful embedding generation and batching."""
    from ragframework.embeddings.openai import OpenAIEmbedder

    mock_client_instance = MagicMock()
    mock_openai.OpenAI.return_value = mock_client_instance

    embedder = OpenAIEmbedder(api_key="fake-key", batch_size=2)
    embedder.client = mock_client_instance

    class MockEmbedding:
        def __init__(self, index, embedding):
            self.index = index
            self.embedding = embedding

    class MockResponse:
        def __init__(self, data):
            self.data = data

    def mock_create(input, model):
        data = [MockEmbedding(index=i, embedding=[float(i), float(i)]) for i in range(len(input))]
        data.reverse()
        return MockResponse(data=data)

    mock_client_instance.embeddings.create.side_effect = mock_create

    texts = ["text1", "text2", "text3"]
    results = embedder.embed(texts)

    assert len(results) == 3
    assert results[0] == [0.0, 0.0]
    assert results[1] == [1.0, 1.0]
    assert results[2] == [0.0, 0.0]

    assert mock_client_instance.embeddings.create.call_count == 2


@patch("ragframework.embeddings.openai.openai")
def test_openai_embedder_embed_error(mock_openai):
    """Test that API errors are wrapped in EmbedderError."""
    from ragframework.embeddings.openai import OpenAIEmbedder

    mock_client_instance = MagicMock()
    mock_openai.OpenAI.return_value = mock_client_instance
    embedder = OpenAIEmbedder(api_key="fake-key")
    embedder.client = mock_client_instance

    mock_client_instance.embeddings.create.side_effect = Exception("API offline")

    with pytest.raises(EmbedderError, match="OpenAI API call failed: API offline"):
        embedder.embed(["text"])
