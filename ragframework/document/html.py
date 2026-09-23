"""HTML document loading without optional parsing dependencies."""

from __future__ import annotations

import math
from html.parser import HTMLParser
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen

from ragframework.base import Document, DocumentLoader
from ragframework.document.loaders import _make_id
from ragframework.exceptions import LoaderError
from ragframework.utils.text import normalize_whitespace

_IGNORED_TAGS = {"head", "nav", "noscript", "script", "style", "template"}
_NON_DOCUMENT_TITLE_TAGS = (_IGNORED_TAGS - {"head"}) | {"svg", "math"}
_HEAD_TAGS = {
    "base",
    "link",
    "meta",
    "noframes",
    "noscript",
    "script",
    "style",
    "template",
    "title",
}
_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "details",
    "div",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "summary",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}


class _TextParser(HTMLParser):
    """Collect visible text and title text while tracking excluded ancestors."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text: list[str] = []
        self.title: list[str] = []
        self._tags: list[str] = []
        self._positions: dict[str, list[int]] = {}
        self._title_seen = False
        self._capture_title = False

    def _is_ignored(self) -> bool:
        return any(tag in self._positions for tag in _IGNORED_TAGS)

    def _close_from(self, index: int) -> None:
        # Each open tag is removed once, including implicitly closed children.
        for tag in reversed(self._tags[index:]):
            positions = self._positions[tag]
            positions.pop()
            if not positions:
                del self._positions[tag]
        del self._tags[index:]
        if "title" not in self._positions:
            self._capture_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # Both </head> and <body> may be omitted before the first body element.
        if self._tags and self._tags[-1] == "head" and tag not in _HEAD_TAGS:
            self._close_from(len(self._tags) - 1)
        if tag in _BLOCK_TAGS and not self._is_ignored():
            self.text.append(" ")
        if tag == "title":
            self._capture_title = not self._title_seen and not any(
                ancestor in self._positions for ancestor in _NON_DOCUMENT_TITLE_TAGS
            )
            if self._capture_title:
                self._title_seen = True
        if tag not in _VOID_TAGS:
            self._positions.setdefault(tag, []).append(len(self._tags))
            self._tags.append(tag)

    def handle_endtag(self, tag: str) -> None:
        positions = self._positions.get(tag)
        if positions:
            self._close_from(positions[-1])
        if tag in _BLOCK_TAGS and not self._is_ignored():
            self.text.append(" ")

    def handle_data(self, data: str) -> None:
        # Non-whitespace text also implicitly ends head. HTML whitespace does
        # not include NBSP or other Unicode whitespace recognized by str.strip.
        if self._tags and self._tags[-1] == "head" and data.strip(" \t\n\f\r"):
            self._close_from(len(self._tags) - 1)
        if self._capture_title and not any(
            tag in self._positions for tag in _NON_DOCUMENT_TITLE_TAGS
        ):
            self.title.append(data)
        elif "title" not in self._positions and not self._is_ignored():
            self.text.append(data)


class HTMLLoader(DocumentLoader):
    """Extract one document from a local HTML file or an HTTP(S) URL.

    Uses the standard-library HTML parser, without JavaScript execution.
    Script, style, navigation, noscript, template, and head content is omitted;
    the first document title is retained separately in metadata (excluding SVG
    and MathML titles). Block boundaries become
    spaces, inline text is preserved, and whitespace is collapsed.

    Args:
        encoding: Local file encoding and fallback for HTTP responses without
            a charset. An HTTP Content-Type charset takes precedence.
        timeout: Positive, finite timeout in seconds for HTTP requests.
        user_agent: User-Agent header sent when fetching a URL.
    """

    def __init__(
        self,
        encoding: str = "utf-8",
        timeout: float = 10.0,
        user_agent: str = "ragframework/HTMLLoader",
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a positive, finite number of seconds.")
        self.encoding = encoding
        self.timeout = timeout
        self.user_agent = user_agent

    def load(self, source: str) -> list[Document]:
        """Read and extract HTML, wrapping file, network, and decoding errors."""
        try:
            if source.lower().startswith(("http://", "https://")):
                request = Request(source, headers={"User-Agent": self.user_agent})
                with urlopen(request, timeout=self.timeout) as response:
                    encoding = response.headers.get_content_charset() or self.encoding
                    markup = response.read().decode(encoding)
            else:
                markup = Path(source).read_text(encoding=self.encoding)
        except (OSError, ValueError, LookupError, HTTPException) as exc:
            raise LoaderError(f"Could not read HTML from {source}: {exc}") from exc

        parser = _TextParser()
        try:
            parser.feed(markup)
            parser.close()
        except (AssertionError, ValueError) as exc:
            raise LoaderError(f"Could not parse HTML from {source}: {exc}") from exc
        return [
            Document(
                id=_make_id(source),
                content=normalize_whitespace("".join(parser.text)),
                metadata={
                    "source": source,
                    "title": normalize_whitespace("".join(parser.title)),
                    "format": "html",
                },
            )
        ]
