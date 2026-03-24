"""Placeholder generator for pipeline testing."""

from __future__ import annotations

from ragframework.base import Chunk, Generator


class EchoGenerator(Generator):
    """Return the retrieved context as the answer — for testing only.

    This generator lets you verify the full ingest → retrieve → generate
    pipeline without calling any LLM API. Replace it with an
    :class:`OpenAIGenerator` or :class:`AnthropicGenerator` for real use
    (see ``.github/GOOD_FIRST_ISSUES.md``).
    """

    def generate(self, query: str, context: list[Chunk]) -> str:
        if not context:
            return f"No context found for query: {query!r}"
        parts = [f"[{i + 1}] {chunk.content}" for i, chunk in enumerate(context)]
        return "\n\n".join(parts)
