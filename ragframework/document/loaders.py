"""Built-in document loaders.

These loaders handle plain text and Markdown files with no extra dependencies.
For PDF, DOCX, HTML, and other formats see the open issues in
`.github/GOOD_FIRST_ISSUES.md`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from ragframework.base import Document, DocumentLoader
from ragframework.exceptions import LoaderError


def _make_id(source: str) -> str:
    return hashlib.md5(source.encode()).hexdigest()[:12]


class TextFileLoader(DocumentLoader):
    """Load a plain-text (``.txt``) file as a single :class:`Document`."""

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise LoaderError(f"File not found: {source}")
        if not path.is_file():
            raise LoaderError(f"Not a file: {source}")
        try:
            content = path.read_text(encoding=self.encoding)
        except OSError as exc:
            raise LoaderError(f"Could not read {source}: {exc}") from exc
        return [
            Document(
                id=_make_id(source),
                content=content,
                metadata={"source": source, "filename": path.name},
            )
        ]


class MarkdownLoader(DocumentLoader):
    """Load a Markdown (``.md``) file as a single :class:`Document`.

    The raw Markdown text is stored as-is — no rendering is applied.
    This loader is intentionally minimal; a richer implementation that
    strips front-matter or renders HTML would make a great contribution.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def load(self, source: str) -> list[Document]:
        path = Path(source)
        if not path.exists():
            raise LoaderError(f"File not found: {source}")
        if not path.is_file():
            raise LoaderError(f"Not a file: {source}")
        try:
            content = path.read_text(encoding=self.encoding)
        except OSError as exc:
            raise LoaderError(f"Could not read {source}: {exc}") from exc
        return [
            Document(
                id=_make_id(source),
                content=content,
                metadata={"source": source, "filename": path.name, "format": "markdown"},
            )
        ]
