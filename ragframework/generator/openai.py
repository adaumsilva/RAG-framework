"""OpenAI-powered LLM generator."""

from __future__ import annotations

import os
from typing import Any

from ragframework.base import Chunk, Generator
from ragframework.exceptions import GeneratorError


class OpenAIGenerator(Generator):
    """Generate answers using the OpenAI Chat Completions API.

    Args:
        model: OpenAI chat model to use.
        api_key: OpenAI API key. Falls back to the ``OPENAI_API_KEY``
            environment variable when not provided.
        system_prompt: System instruction used to ground the answer in
            retrieved context.
        max_tokens: Maximum number of tokens in the generated response.

    Raises:
        GeneratorError: If the OpenAI client cannot be initialized or the
            API request fails.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        system_prompt: str = (
            "You are a helpful assistant. Answer the user's question using "
            "the provided context. If the context does not contain enough "
            "information to answer, say so clearly."
        ),
        max_tokens: int = 512,
    ) -> None:

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI support requires 'ragframework[openai]'. "
                "Install it with: pip install ragframework[openai]"
            ) from exc

        resolved_api_key = (api_key or os.environ.get("OPENAI_API_KEY") or "").strip()

        if not resolved_api_key:
            raise GeneratorError(
                "No API key provided. Pass `api_key` or set the "
                "`OPENAI_API_KEY` environment variable."
            )

        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

        try:
            self._client: Any = OpenAI(api_key=resolved_api_key)
        except Exception as exc:
            raise GeneratorError(f"Could not initialize OpenAI client: {exc}") from exc

    def generate(self, query: str, context: list[Chunk]) -> str:
        """Generate an answer from the query and retrieved context.

        Args:
            query: The user's question.
            context: Retrieved chunks used to ground the answer.

        Returns:
            The generated answer text.

        Raises:
            GeneratorError: If the OpenAI API request fails or returns
                an unexpected response.
        """
        formatted_context = self._format_context(context)

        user_prompt = f"Context:\n{formatted_context}\n\nQuestion:\n{query}"

        try:
            response: Any = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
            )

            answer = response.choices[0].message.content

            if not answer:
                raise GeneratorError("OpenAI returned an empty response.")

            return str(answer)

        except GeneratorError:
            raise
        except Exception as exc:
            raise GeneratorError(f"OpenAI generation failed: {exc}") from exc

    @staticmethod
    def _format_context(context: list[Chunk]) -> str:
        """Format retrieved chunks as numbered context."""
        if not context:
            return "No context was retrieved."

        return "\n\n".join(
            f"[Chunk {index}]\n{chunk.content}" for index, chunk in enumerate(context, start=1)
        )
