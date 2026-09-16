"""OpenAI embeddings integration."""

from __future__ import annotations

import os
from typing import Any

from ragframework.base import Embedder
from ragframework.exceptions import EmbedderError


class OpenAIEmbedder(Embedder):
    """Produce embeddings using the OpenAI API.

    Requires the `openai` Python package. Install it with:
        pip install "ragframework[openai]"

    Args:
        model: The OpenAI embedding model to use (default: "text-embedding-3-small").
        api_key: The OpenAI API key. If not provided, it falls back to
            the `OPENAI_API_KEY` environment variable.
        batch_size: The maximum number of texts to send in a single API request
            (default: 2048, which is a safe limit for OpenAI embeddings).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        batch_size: int = 2048,
    ) -> None:
        try:
            import openai
        except ImportError as exc:
            raise EmbedderError(
                "The `openai` package is not installed. "
                'Please install it with `pip install "ragframework[openai]"`.'
            ) from exc

        self.model = model
        self.batch_size = batch_size

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise EmbedderError(
                "No API key provided. Pass `api_key` or set the "
                "`OPENAI_API_KEY` environment variable."
            )

        self.client: Any = openai.OpenAI(api_key=key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                response = self.client.embeddings.create(input=batch, model=self.model)
            except Exception as e:
                raise EmbedderError(f"OpenAI API call failed: {e}") from e

            # The response.data list is guaranteed to be in the same order as the input
            # We sort it just to be safe, as per OpenAI's typical data structure,
            # though it's usually already ordered.
            sorted_data = sorted(response.data, key=lambda x: x.index)
            results.extend(item.embedding for item in sorted_data)

        return results
