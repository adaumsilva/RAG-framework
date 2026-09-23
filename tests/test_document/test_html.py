"""HTML loading tests using local files and mocked HTTP responses."""

from email.message import Message
from http.client import IncompleteRead
from io import BytesIO
from urllib.error import HTTPError, URLError

import pytest

from ragframework.base import DocumentLoader
from ragframework.document import HTMLLoader
from ragframework.document.loaders import _make_id
from ragframework.exceptions import LoaderError


def test_local_page_extracts_text_title_and_metadata(tmp_path):
    path = tmp_path / "article.html"
    path.write_text(
        """<!doctype html><html><head>
        <meta charset="utf-8"><title>  A &amp; B — Guide </title>
        <style>p {color: red}</style><script>window.title = 'hidden';</script>
        </head><body><nav><ul><li><a href="/">Menu</a></li></ul></nav>
        <article><h1>Article</h1><p>A <b>bold</b> word.<br>Next line.</p>
        <ul><li>Item one</li><li>Item two</li></ul></article>
        <noscript>Enable JavaScript</noscript><template><div>Hidden template</div></template>
        <script>const example = '<p>Hidden markup</p>';</script><!-- Hidden comment -->
        </body></html>""",
        encoding="utf-8",
    )
    loader = HTMLLoader()

    docs = loader.load(str(path))

    assert isinstance(loader, DocumentLoader)
    assert len(docs) == 1
    assert docs[0].id == _make_id(str(path))
    assert docs[0].content == "Article A bold word. Next line. Item one Item two"
    assert docs[0].metadata == {"source": str(path), "title": "A & B — Guide", "format": "html"}
    assert loader.load(str(path)) == docs


@pytest.mark.parametrize(
    ("markup", "content", "title"),
    [
        ("", "", ""),
        ("<title>Only a title</title>", "", "Only a title"),
        ("<p>Hello <b>world</b>! co<em>operate</em>.</p>", "Hello world! cooperate.", ""),
        ("<p>One</p><div>Two</div><p>Three<br/>Four</p>", "One Two Three Four", ""),
        ("<p>A&nbsp;B &#38; C\n\tD</p>", "A B & C D", ""),
        ("<p>a<template><div>Hidden</div></template>b</p>", "ab", ""),
        ("<nav><nav>Nested</nav>Still hidden</nav><p>Visible</p>", "Visible", ""),
        ("<template><title>Hidden title</title></template><p>Visible</p>", "Visible", ""),
        ("<head><meta charset='utf-8'><title>Title</title><body><p>Body", "Body", "Title"),
        ("<head><title>Title</title><h1>Hello</h1><p>World</p>", "Hello World", "Title"),
        ("<head><title>Title</title>Readable body text", "Readable body text", "Title"),
        ("<head><title>Title</title>Readable <b>body</b> text", "Readable body text", "Title"),
        ("<head><title>Title</title>&amp; body text", "& body text", "Title"),
        ("<head><title>Title</title>&nbsp;body text", "body text", "Title"),
        ("<head><title>Title</title>\vbody text", "body text", "Title"),
        (
            "<head>\n \t<meta charset='utf-8'><title>Title</title>\r\n</head><p>Body</p>",
            "Body",
            "Title",
        ),
        (
            "<head><title>Main</title></head><svg><title>Icon</title><text>Legend</text></svg>",
            "Legend",
            "Main",
        ),
        ("<svg><title>Icon</title></svg><title>Page</title><p>Text</p>", "Text", "Page"),
        ("<svg><title>Icon</title></svg><p>Text</p>", "Text", ""),
        ("<math><title>Equation</title></math><p>Text</p>", "Text", ""),
        ("<title>First</title><title>Second</title><p>Text</p>", "Text", "First"),
        ("<title></title><title>Second</title><p>Text</p>", "Text", ""),
        (
            "<ul><li>Outer<ul><li>Inner<li>Sibling</ul><li>After</ul>",
            "Outer Inner Sibling After",
            "",
        ),
        ("<ul><li><b>One</b><li><b>Two</b><li>Three</ul><p>After</p>", "One Two Three After", ""),
        ("<head><template><body>Hidden</body></template></head><p>Visible</p>", "Visible", ""),
        ("<head><link href='x'><meta charset='utf-8'></head><p>Visible</p>", "Visible", ""),
        ("<nav/><p>A<img src='x'>B<input/>C</p>", "ABC", ""),
        ("<p>Text</p></unknown><p>After</p>", "Text After", ""),
        ("<p>First<p>Second", "First Second", ""),
        (
            "<TABLE><TR><TH>Name</TH><TH>Value</TH></TR><TR><TD>A</TD><TD>B</TD></TR></TABLE>",
            "Name Value A B",
            "",
        ),
    ],
)
def test_html_fragments(tmp_path, markup, content, title):
    path = tmp_path / "fragment.html"
    path.write_text(markup, encoding="utf-8")
    doc = HTMLLoader().load(str(path))[0]
    assert doc.content == content
    assert doc.metadata["title"] == title


def test_local_encoding_can_be_configured(tmp_path):
    path = tmp_path / "latin.html"
    path.write_text("<title>Café</title><p>Crème</p>", encoding="latin-1")
    doc = HTMLLoader(encoding="latin-1").load(str(path))[0]
    assert doc.content == "Crème"
    assert doc.metadata["title"] == "Café"


def test_reusing_loader_does_not_retain_previous_page(tmp_path):
    first = tmp_path / "first.html"
    second = tmp_path / "second.html"
    first.write_text("<title>First</title><p>First page</p>", encoding="utf-8")
    second.write_text("<p>Second page</p>", encoding="utf-8")
    loader = HTMLLoader()
    loader.load(str(first))
    doc = loader.load(str(second))[0]
    assert doc.content == "Second page"
    assert doc.metadata["title"] == ""
    assert doc.id == _make_id(str(second))


def test_missing_file_and_directory_are_wrapped(tmp_path):
    for path in (tmp_path / "missing.html", tmp_path):
        with pytest.raises(LoaderError, match="Could not read HTML") as exc:
            HTMLLoader().load(str(path))
        assert isinstance(exc.value.__cause__, OSError)


def test_decode_error_is_wrapped(tmp_path):
    path = tmp_path / "invalid.html"
    path.write_bytes(b"<p>\xff</p>")
    with pytest.raises(LoaderError, match="Could not read HTML") as exc:
        HTMLLoader().load(str(path))
    assert isinstance(exc.value.__cause__, UnicodeDecodeError)


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_url_uses_timeout_user_agent_and_response_charset(monkeypatch, scheme):
    class Response(BytesIO):
        headers = Message()

    response = Response("<title>Café</title><p>Crème</p>".encode("latin-1"))
    response.headers["Content-Type"] = "text/html; charset=iso-8859-1"
    source = f"{scheme}://example.com/article"

    def fake_urlopen(request, *, timeout):
        assert request.full_url == source
        assert request.get_header("User-agent") == "test-rag/1.0"
        assert timeout == 2.5
        return response

    monkeypatch.setattr("ragframework.document.html.urlopen", fake_urlopen)
    doc = HTMLLoader(timeout=2.5, user_agent="test-rag/1.0").load(source)[0]
    assert doc.content == "Crème"
    assert doc.metadata == {"source": source, "title": "Café", "format": "html"}
    assert doc.id == _make_id(source)
    assert response.closed


@pytest.mark.parametrize("encoding", ["utf-8", "latin-1"])
def test_url_without_charset_uses_configured_encoding(monkeypatch, encoding):
    class Response(BytesIO):
        headers = Message()

    response = Response("<p>Café</p>".encode(encoding))
    monkeypatch.setattr("ragframework.document.html.urlopen", lambda *args, **kwargs: response)
    doc = HTMLLoader(encoding=encoding).load("https://example.com/page")[0]
    assert doc.content == "Café"
    assert response.closed


@pytest.mark.parametrize(
    "error",
    [
        URLError("unavailable"),
        HTTPError("https://example.com/page", 404, "Not found", Message(), None),
        TimeoutError("timed out"),
        IncompleteRead(b"partial"),
    ],
)
def test_network_errors_are_wrapped(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr("ragframework.document.html.urlopen", fail)
    with pytest.raises(
        LoaderError, match="Could not read HTML from https://example.com/page"
    ) as exc:
        HTMLLoader().load("https://example.com/page")
    assert exc.value.__cause__ is error


@pytest.mark.parametrize("charset", ["utf-8", "not-a-real-encoding"])
def test_url_decoding_errors_are_wrapped_and_response_closed(monkeypatch, charset):
    class Response(BytesIO):
        headers = Message()

    response = Response(b"<p>\xff</p>")
    response.headers["Content-Type"] = f"text/html; charset={charset}"
    monkeypatch.setattr("ragframework.document.html.urlopen", lambda *args, **kwargs: response)
    with pytest.raises(LoaderError, match="Could not read HTML"):
        HTMLLoader().load("https://example.com/page")
    assert response.closed


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_timeout_must_be_positive_and_finite(timeout):
    with pytest.raises(ValueError, match="timeout"):
        HTMLLoader(timeout=timeout)
