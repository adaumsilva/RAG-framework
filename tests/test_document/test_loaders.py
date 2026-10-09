"""Tests for built-in document loaders."""

import builtins
import sys
import types
from pathlib import Path

import pytest

from ragframework.base import Document, DocumentLoader
from ragframework.document.loaders import (
    DirectoryLoader,
    DocxLoader,
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

    def test_invalid_encoding_raises_loader_error(self, tmp_path):
        path = tmp_path / "invalid.txt"
        path.write_bytes(b"\xff\xfe")

        loader = TextFileLoader(encoding="utf-8")

        with pytest.raises(
            LoaderError,
            match=r"Could not decode .*invalid\.txt.*utf-8",
        ):
            loader.load(str(path))

    def test_unknown_encoding_raises_loader_error(self, tmp_path):
        path = tmp_path / "document.txt"
        path.write_text("hello", encoding="utf-8")

        loader = TextFileLoader(encoding="nope")

        with pytest.raises(
            LoaderError,
            match=r"Unknown encoding 'nope'.*document\.txt",
        ):
            loader.load(str(path))

    @pytest.mark.parametrize(
        ("errors", "expected"),
        [
            ("replace", "\ufffd"),
            ("ignore", ""),
        ],
    )
    def test_error_handling(self, tmp_path, errors, expected):
        path = tmp_path / "invalid.txt"
        path.write_bytes(b"Hello \xff world")

        loader = TextFileLoader(encoding="utf-8", errors=errors)
        docs = loader.load(str(path))

        assert len(docs) == 1
        assert expected in docs[0].content


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

    def test_invalid_encoding_raises_loader_error(self, tmp_path):
        path = tmp_path / "invalid.md"
        path.write_bytes(b"\xff\xfe")

        loader = MarkdownLoader(encoding="utf-8")

        with pytest.raises(
            LoaderError,
            match=r"Could not decode .*invalid\.md.*utf-8",
        ):
            loader.load(str(path))

    def test_unknown_encoding_raises_loader_error(self, tmp_path):
        path = tmp_path / "document.md"
        path.write_text("# Hello", encoding="utf-8")

        loader = MarkdownLoader(encoding="nope")

        with pytest.raises(
            LoaderError,
            match=r"Unknown encoding 'nope'.*document\.md",
        ):
            loader.load(str(path))

    @pytest.mark.parametrize(
        ("errors", "expected"),
        [
            ("replace", "\ufffd"),
            ("ignore", ""),
        ],
    )
    def test_error_handling(self, tmp_path, errors, expected):
        path = tmp_path / "invalid.md"
        path.write_bytes(b"# Hello \xff world")

        loader = MarkdownLoader(encoding="utf-8", errors=errors)
        docs = loader.load(str(path))

        assert len(docs) == 1
        assert expected in docs[0].content


class TestDocxLoader:
    def test_loads_paragraphs(self, tmp_path):
        from docx import Document as DocxDocument

        docx_path = tmp_path / "doc.docx"

        doc = DocxDocument()
        doc.add_paragraph("First paragraph")
        doc.add_paragraph("")
        doc.add_paragraph("Second paragraph")
        doc.save(docx_path)

        loader = DocxLoader()
        docs = loader.load(str(docx_path))

        assert len(docs) == 2
        assert docs[0].content == "First paragraph"
        assert docs[1].content == "Second paragraph"
        assert docs[0].metadata["format"] == "docx"
        assert docs[0].metadata["paragraph_number"] == 1
        assert docs[1].metadata["paragraph_number"] == 2

    def test_loads_whole_file(self, tmp_path):
        from docx import Document as DocxDocument

        docx_path = tmp_path / "doc.docx"

        doc = DocxDocument()
        doc.add_paragraph("First paragraph")
        doc.add_paragraph("Second paragraph")
        doc.save(docx_path)

        loader = DocxLoader(split_paragraphs=False)
        docs = loader.load(str(docx_path))

        assert len(docs) == 1
        assert "First paragraph" in docs[0].content
        assert "Second paragraph" in docs[0].content
        assert docs[0].metadata["format"] == "docx"
        assert docs[0].metadata["split_paragraphs"] is False

    def test_missing_file_raises(self):
        loader = DocxLoader()

        with pytest.raises(LoaderError, match="File not found"):
            loader.load("/nonexistent/path/file.docx")

    def test_missing_dependency_raises(self, tmp_path, monkeypatch):
        docx_path = tmp_path / "doc.docx"
        docx_path.write_bytes(b"fake docx content")

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "docx":
                raise ImportError("No module named 'docx'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        loader = DocxLoader()

        with pytest.raises(
            LoaderError,
            match=r"DOCX support requires 'ragframework\[docx\]'",
        ):
            loader.load(str(docx_path))

    def test_invalid_docx_raises(self, tmp_path):
        docx_path = tmp_path / "invalid.docx"
        docx_path.write_text("This is not a valid DOCX file.", encoding="utf-8")

        loader = DocxLoader()

        with pytest.raises(LoaderError, match="Could not read DOCX file"):
            loader.load(str(docx_path))


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

def test_directory_loader_non_recursive_ignores_nested_files(tmp_path):
    docs_dir = tmp_path / "docs"
    nested_dir = docs_dir / "nested"
    nested_dir.mkdir(parents=True)

    (docs_dir / "top.txt").write_text("Top", encoding="utf-8")
    (nested_dir / "deep.txt").write_text("Deep", encoding="utf-8")

    loader = DirectoryLoader(
        loaders={".txt": TextFileLoader()},
        recursive=False,
    )

    docs = loader.load(str(docs_dir))

    assert [doc.metadata["relative_path"] for doc in docs] == ["top.txt"]

def test_directory_loader_default_loaders_skip_pdf_without_pypdf(monkeypatch):
    monkeypatch.setattr(
        "ragframework.document.loaders.importlib.util.find_spec",
        lambda name: None,
    )

    loader = DirectoryLoader()

    assert ".pdf" not in loader.loaders
    assert ".txt" in loader.loaders

def test_directory_loader_counts_skipped_unsupported_files(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "keep.txt").write_text("Keep", encoding="utf-8")
    (docs_dir / "ignored.py").write_text("print('x')", encoding="utf-8")
    (docs_dir / "ignored.csv").write_text("a,b", encoding="utf-8")

    loader = DirectoryLoader(loaders={".txt": TextFileLoader()})

    docs = loader.load(str(docs_dir))

    assert len(docs) == 1
    assert loader.skipped_count == 2

def test_directory_loader_resets_skipped_count_between_loads(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "keep.txt").write_text("Keep", encoding="utf-8")
    (docs_dir / "ignored.py").write_text("print('x')", encoding="utf-8")

    loader = DirectoryLoader(loaders={".txt": TextFileLoader()})

    loader.load(str(docs_dir))
    assert loader.skipped_count == 1

    loader.load(str(docs_dir))
    assert loader.skipped_count == 1

    (docs_dir / "ignored.py").unlink()

    loader.load(str(docs_dir))
    assert loader.skipped_count == 0
