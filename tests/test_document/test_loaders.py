"""Tests for built-in document loaders."""

import builtins
import sys
import types

import pytest

from ragframework.document.loaders import MarkdownLoader, PDFLoader, TextFileLoader
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
