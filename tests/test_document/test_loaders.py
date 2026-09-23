"""Tests for built-in document loaders."""

import builtins
import sys
import types
from pathlib import Path

import pytest

from ragframework.base import Document, DocumentLoader
from ragframework.document.loaders import (
    DirectoryLoader,
    MarkdownLoader,
    PDFLoader,
    TextFileLoader,
)
from ragframework.exceptions import LoaderError


class TestTextFileLoader:
    def test_loads_file(self, tmp_text_file):
        loader = TextFileLoader()
        docs = loader.load(tmp_text_file)
        assert len(docs) == 1
        assert "Hello world" in docs[0].content

    def test_metadata_contains_source(self, tmp_text_file):
        loader = TextFileLoader()
        docs = loader.load(tmp_text_file)
        assert docs[0].metadata["source"] == tmp_text_file

    def test_missing_file_raises(self):
        loader = TextFileLoader()
        with pytest.raises(LoaderError, match="not found"):
            loader.load("/nonexistent/path/file.txt")

    def test_directory_raises(self, tmp_path):
        loader = TextFileLoader()
        with pytest.raises(LoaderError, match="Not a file"):
            loader.load(str(tmp_path))


class TestMarkdownLoader:
    def test_loads_markdown(self, tmp_path):
        md = tmp_path / "doc.md"
        md.write_text("# Title\n\nSome content.", encoding="utf-8")
        loader = MarkdownLoader()
        docs = loader.load(str(md))
        assert len(docs) == 1
        assert "# Title" in docs[0].content
        assert docs[0].metadata["format"] == "markdown"

    def test_missing_file_raises(self):
        loader = MarkdownLoader()
        with pytest.raises(LoaderError):
            loader.load("/no/such/file.md")


def test_pdf_loader_requires_pypdf(tmp_path, monkeypatch):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")

    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pypdf":
            raise ImportError("No module named 'pypdf'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    loader = PDFLoader()
    with pytest.raises(ImportError, match=r"PDF support requires 'ragframework\[pdf\]'"):
        loader.load(str(pdf_path))


def test_pdf_loader_loads_pages_with_fake_pypdf(tmp_path, monkeypatch):
    class FakePage:
        def __init__(self, text):
            self._text = text

        def extract_text(self):
            return self._text

    class FakePdfReader:
        def __init__(self, source):
            self.pages = [FakePage("page1"), FakePage("page2")]

    fake_pypdf = types.SimpleNamespace(PdfReader=FakePdfReader)
    monkeypatch.setitem(sys.modules, "pypdf", fake_pypdf)

    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%EOF\n")

    loader = PDFLoader(split_pages=True)
    docs = loader.load(str(pdf_path))

    assert len(docs) == 2
    assert docs[0].content == "page1"
    assert docs[1].content == "page2"

    loader_whole = PDFLoader(split_pages=False)
    docs_whole = loader_whole.load(str(pdf_path))
    assert len(docs_whole) == 1
    assert "page1" in docs_whole[0].content
    assert "page2" in docs_whole[0].content

def test_directory_loader_loads_matching_files(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "hello.txt").write_text(
        "Hello from text.",
        encoding="utf-8",
    )

    (docs_dir / "readme.md").write_text(
        "# README",
        encoding="utf-8",
    )

    (docs_dir / "ignored.py").write_text(
        "print('ignored')",
        encoding="utf-8",
    )

    loader = DirectoryLoader(
        loaders={
            ".txt": TextFileLoader(),
            ".md": MarkdownLoader(),
        }
    )

    docs = loader.load(str(docs_dir))

    assert len(docs) == 2
    assert docs[0].metadata["relative_path"] == "hello.txt"
    assert docs[1].metadata["relative_path"] == "readme.md"

def test_directory_loader_loads_nested_files_in_sorted_order(tmp_path):
    docs_dir = tmp_path / "docs"
    nested_dir = docs_dir / "nested"
    nested_dir.mkdir(parents=True)

    (docs_dir / "z.txt").write_text("Z", encoding="utf-8")
    (docs_dir / "a.txt").write_text("A", encoding="utf-8")
    (nested_dir / "b.txt").write_text("B", encoding="utf-8")

    loader = DirectoryLoader(
        loaders={
            ".txt": TextFileLoader(),
        }
    )

    docs = loader.load(str(docs_dir))

    assert len(docs) == 3

    assert [
        doc.metadata["relative_path"]
        for doc in docs
    ] == [
        "a.txt",
        "nested/b.txt",
        "z.txt",
    ]

def test_directory_loader_skips_loader_errors(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    good_file = docs_dir / "good.txt"
    bad_file = docs_dir / "bad.txt"

    good_file.write_text("Good", encoding="utf-8")
    bad_file.write_text("Bad", encoding="utf-8")

    class FailingLoader(DocumentLoader):
        def load(self, source: str) -> list[Document]:
            raise LoaderError(f"Cannot load {source}")

    loader = DirectoryLoader(
        loaders={".txt": FailingLoader()},
        on_error="skip",
    )

    docs = loader.load(str(docs_dir))

    assert docs == []

def test_directory_loader_raises_loader_error(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    bad_file = docs_dir / "bad.txt"
    bad_file.write_text("Bad", encoding="utf-8")

    class FailingLoader(DocumentLoader):
        def load(self, source: str) -> list[Document]:
            raise LoaderError(f"Cannot load {source}")

    loader = DirectoryLoader(
        loaders={".txt": FailingLoader()},
        on_error="raise",
    )

    with pytest.raises(LoaderError, match="bad.txt"):
        loader.load(str(docs_dir))

def test_directory_loader_has_default_loaders(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "notes.txt").write_text(
        "Some notes",
        encoding="utf-8",
    )

    (docs_dir / "readme.md").write_text(
        "# README",
        encoding="utf-8",
    )

    (docs_dir / "guide.markdown").write_text(
        "# Guide",
        encoding="utf-8",
    )

    loader = DirectoryLoader()

    docs = loader.load(str(docs_dir))

    assert len(docs) == 3
    assert {
        doc.metadata["relative_path"]
        for doc in docs
    } == {
        "notes.txt",
        "readme.md",
        "guide.markdown",
    }

def test_directory_loader_rejects_invalid_on_error(tmp_path):
    with pytest.raises(ValueError, match="on_error"):
        DirectoryLoader(on_error="banana")

def test_directory_loader_handles_unreadable_file(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    good_file = docs_dir / "good.txt"
    bad_file = docs_dir / "bad.txt"

    good_file.write_text("Good", encoding="utf-8")
    bad_file.write_text("Bad", encoding="utf-8")

    original_read_text = Path.read_text

    def fake_read_text(self, *args, **kwargs):
        if self == bad_file:
            raise OSError("Permission denied")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    skip_loader = DirectoryLoader(on_error="skip")
    docs = skip_loader.load(str(docs_dir))

    assert len(docs) == 1
    assert docs[0].metadata["relative_path"] == "good.txt"

    raise_loader = DirectoryLoader(on_error="raise")

    with pytest.raises(LoaderError, match="bad.txt"):
        raise_loader.load(str(docs_dir))
