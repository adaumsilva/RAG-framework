"""Minimal RAG generation example using OpenAI."""

import os

from ragframework.base import Chunk
from ragframework.generator import OpenAIGenerator

if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("Set OPENAI_API_KEY before running this example.")

generator = OpenAIGenerator()
context = [
    Chunk(
        id="example-1",
        content="Retrieval-Augmented Generation retrieves relevant documents "
        "and supplies them to an LLM as context.",
    )
]

print(generator.generate("What is RAG?", context))
