"""Anthropic Claude generator for grounded RAG answers."""

from __future__ import annotations

import os
from typing import Any

from ragframework.base import Chunk, Generator
from ragframework.exceptions import GeneratorError

_INSTALL_HINT = (
    "Anthropic support requires the 'anthropic' package. "
    "Install it with: pip install ragframework[anthropic]"
)

_DEFAULT_SYSTEM_PROMPT = (
    "Answer the question using only the provided context. If the context is insufficient, say so."
)


class AnthropicGenerator(Generator):
    """Generate answers using the Anthropic Messages API.

    Args:
        model: Anthropic model identifier (for example ``"claude-sonnet-5"``).
        api_key: API key. Falls back to ``ANTHROPIC_API_KEY`` when omitted.
        system_prompt: System instruction passed to the Messages API.
        max_tokens: Maximum tokens to generate.

    Raises:
        ImportError: If the ``anthropic`` optional dependency is not installed.
        GeneratorError: If the API key is missing or generation fails.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        max_tokens: int = 1024,
    ) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc

        resolved_key = (api_key or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not resolved_key:
            raise GeneratorError(
                "No API key provided. Pass `api_key` or set the "
                "`ANTHROPIC_API_KEY` environment variable."
            )

        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        try:
            self._client: Any = Anthropic(api_key=resolved_key)
        except Exception as exc:
            raise GeneratorError("Could not create Anthropic client.") from exc

    def generate(self, query: str, context: list[Chunk]) -> str:
        """Generate an answer for *query* using retrieved *context* chunks."""
        user_message = self._build_user_message(query, context)
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except Exception as exc:
            raise GeneratorError("Anthropic API call failed.") from exc

        try:
            return self._extract_text(response)
        except GeneratorError:
            raise
        except Exception as exc:
            raise GeneratorError("Could not read Anthropic response.") from exc

    @staticmethod
    def _build_user_message(query: str, context: list[Chunk]) -> str:
        if context:
            numbered = "\n\n".join(
                f"[{index + 1}] {chunk.content}" for index, chunk in enumerate(context)
            )
            return (
                "Use the following context to answer the question.\n\n"
                f"Context:\n{numbered}\n\n"
                f"Question: {query}"
            )
        return f"Use the following context to answer the question.\n\nContext:\n\nQuestion: {query}"

    @staticmethod
    def _extract_text(response: Any) -> str:
        content = getattr(response, "content", None)
        if not content:
            raise GeneratorError("Anthropic response contained no content blocks.")

        text_parts: list[str] = []
        for block in content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", None)
                if isinstance(text, str) and text:
                    text_parts.append(text)

        if not text_parts:
            raise GeneratorError("Anthropic response contained no text blocks.")

        return "".join(text_parts)
