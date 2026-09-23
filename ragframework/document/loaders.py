"""Built-in document loaders.

These loaders handle plain text and Markdown files, PDF with no extra dependencies.
For DOCX, HTML, and other formats see the open issues in
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


class PDFLoader(DocumentLoader):
    """Load a PDF file into one or more :class:`Document` objects.
    By default, each page is loaded as a separate document with metadata indicating the page number.
    To load the entire PDF as a single document, set ``split_pages=False``
    """

    def __init__(self, split_pages: bool = True) -> None:
        self.split_pages = split_pages

    def load(self, source: str) -> list[Document]:
        path = Path(source)

        if not path.exists():
            raise LoaderError(f"File not found: {source}")
        if not path.is_file():
            raise LoaderError(f"Not a file: {source}")

        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "PDF support requires 'ragframework[pdf]'. "
                "Install it with: pip install ragframework[pdf]"
            ) from exc

        # Open the PDF
        try:
            reader = PdfReader(str(path))
        except Exception as exc:
            raise LoaderError(f"Could not read PDF file {source}: {exc}") from exc

        documents: list[Document] = []

        if not self.split_pages:
            # One single Document for the whole PDF
            try:
                full_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as exc:
                raise LoaderError(f"Failed to extract text from {source}: {exc}") from exc

            documents.append(
                Document(
                    id=_make_id(source),
                    content=full_text.strip(),
                    metadata={
                        "source": source,
                        "filename": path.name,
                        "format": "pdf",
                        "total_pages": len(reader.pages),
                        "split_pages": False,
                    },
                )
            )
        else:
            # One Document per page (default)
            for i, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""
                except Exception as exc:
                    raise LoaderError(
                        f"Failed to extract text from page {i} of {source}: {exc}"
                    ) from exc

                documents.append(
                    Document(
                        id=_make_id(f"{source}_page{i}"),
                        content=text,
                        metadata={
                            "source": source,
                            "filename": path.name,
                            "format": "pdf",
                            "page_number": i,
                            "total_pages": len(reader.pages),
                        },
                    )
                )
        return documents


class DirectoryLoader(DocumentLoader):
    """Load documents from a directory using loaders selected by file extension."""

    def __init__(
        self,
        loaders: dict[str, DocumentLoader] | None = None,
        glob: str = "**/*",
        recursive: bool = True,
        on_error: str = "raise",
    ) -> None:
        if on_error not in {"skip", "raise"}:
            raise ValueError("on_error must be either 'skip' or 'raise'")

        if loaders is None:
            loaders = {
                ".txt": TextFileLoader(),
                ".md": MarkdownLoader(),
                ".markdown": MarkdownLoader(),
                ".pdf": PDFLoader(),
            }

        self.loaders = loaders
        self.glob = glob
        self.recursive = recursive
        self.on_error = on_error

    def load(self, source: str) -> list[Document]:
        root = Path(source)

        if not root.exists():
            raise LoaderError(f"Directory not found: {source}")

        if not root.is_dir():
            raise LoaderError(f"Not a directory: {source}")

        paths = root.rglob(self.glob) if self.recursive else root.glob(self.glob)

        documents: list[Document] = []

        for path in sorted(paths):
            if not path.is_file():
                continue

            extension = path.suffix.lower()
            loader = self.loaders.get(extension)

            if loader is None:
                continue

            try:
                loaded_documents = loader.load(str(path))
            except Exception as exc:
                if self.on_error == "skip":
                    continue

                raise LoaderError(
                    f"Failed to load {path}: {exc}"
                ) from exc

            relative_path = path.relative_to(root).as_posix()

            for document in loaded_documents:
                document.metadata["relative_path"] = relative_path
                documents.append(document)

        return documents
