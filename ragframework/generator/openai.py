"""OpenAI chat-completion generator for RAG pipelines."""

from __future__ import annotations

import os

from ragframework.base import Chunk, Generator
from ragframework.exceptions import GeneratorError


class OpenAIGenerator(Generator):
    """Generate grounded answers with the OpenAI Chat Completions API."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        system_prompt: str = (
            "You are a helpful assistant. Answer the user's question using only "
            "the provided context. If the context does not contain the answer, "
            "say that you do not have enough information."
        ),
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

    def generate(self, query: str, context: list[Chunk]) -> str:
        if not self.api_key:
            raise GeneratorError(
                "OpenAI API key is required. Pass api_key or set OPENAI_API_KEY."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise GeneratorError(
                "OpenAI support requires 'ragframework[openai]'. "
                "Install it with: pip install ragframework[openai]"
            ) from exc

        context_text = "\n\n".join(
            f"[{index}] {chunk.content}" for index, chunk in enumerate(context, start=1)
        )
        user_prompt = (
            f"Context:\n{context_text or '(no context retrieved)'}"
            f"\n\nQuestion:\n{query}"
        )

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.max_tokens,
            )
            answer = response.choices[0].message.content
        except Exception as exc:
            raise GeneratorError(f"OpenAI generation failed: {exc}") from exc

        if not answer:
            raise GeneratorError("OpenAI returned an empty response.")

        return answer
