"""CSV and JSON Lines document loaders using only the standard library."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ragframework.base import Document, DocumentLoader
from ragframework.document.loaders import _make_id
from ragframework.exceptions import LoaderError


def _document_from_record(
    record: Mapping[str, Any],
    source: str,
    row_index: int,
    location: str,
    content_keys: list[str],
    metadata_keys: list[str],
    id_key: str | None,
    separator: str,
) -> Document:
    required = content_keys + metadata_keys + ([id_key] if id_key is not None else [])
    for key in required:
        if key not in record:
            raise LoaderError(f"Missing field '{key}' at {location} in {source}.")

    parts: list[str] = []
    for key in content_keys:
        value = record[key]
        if not isinstance(value, str):
            raise LoaderError(f"Content field '{key}' at {location} in {source} must be a string.")
        parts.append(value)

    document_id = f"{_make_id(source)}:{row_index}"
    if id_key is not None:
        value = record[id_key]
        if not isinstance(value, str | int) or isinstance(value, bool):
            raise LoaderError(
                f"ID field '{id_key}' at {location} in {source} must be a string or integer."
            )
        document_id = str(value)

    metadata = {"source": source, **{key: record[key] for key in metadata_keys}}
    metadata["row_index"] = row_index
    return Document(id=document_id, content=separator.join(parts), metadata=metadata)


class CSVLoader(DocumentLoader):
    """Load each CSV data row as a document, using the first row as the header.

    Args:
        content_columns: Columns to join, in order, to form document content.
        metadata_columns: Additional columns to copy into metadata. Other
            columns are omitted. ``row_index`` is reserved for the zero-based
            data row index; selecting ``source`` overrides the source path.
        id_column: Column to use as the document ID. Otherwise, IDs use the
            source path hash and zero-based data row index.
        separator: Text between content columns (a newline by default).
        encoding: File encoding.
        delimiter: Single-character CSV field delimiter.

    Missing fields report one-based data row numbers, excluding the header.
    Quoted delimiters and multiline fields follow standard CSV parsing rules.
    Empty files and header-only files produce no documents.
    """

    def __init__(
        self,
        content_columns: list[str],
        metadata_columns: list[str] | None = None,
        id_column: str | None = None,
        *,
        separator: str = "\n",
        encoding: str = "utf-8",
        delimiter: str = ",",
    ) -> None:
        if not content_columns:
            raise ValueError("content_columns must not be empty.")
        if len(delimiter) != 1 or delimiter in "\r\n":
            raise ValueError("delimiter must be a single character other than a newline.")
        if metadata_columns and "row_index" in metadata_columns:
            raise ValueError("row_index is reserved for the data row index.")
        self.content_columns = list(content_columns)
        self.metadata_columns = list(metadata_columns or [])
        self.id_column = id_column
        self.separator = separator
        self.encoding = encoding
        self.delimiter = delimiter

    def load(self, source: str) -> list[Document]:
        """Read CSV rows, raising ``LoaderError`` for file or record errors."""
        documents: list[Document] = []
        try:
            with Path(source).open(encoding=self.encoding, newline="") as stream:
                reader = csv.DictReader(stream, delimiter=self.delimiter, strict=True)
                for row_index, row in enumerate(reader):
                    if None in row:
                        raise LoaderError(f"Extra CSV fields at row {row_index + 1} in {source}.")
                    # DictReader fills missing trailing fields with None.
                    record = {key: value for key, value in row.items() if value is not None}
                    documents.append(
                        _document_from_record(
                            record,
                            source,
                            row_index,
                            f"row {row_index + 1}",
                            self.content_columns,
                            self.metadata_columns,
                            self.id_column,
                            self.separator,
                        )
                    )
        except csv.Error as exc:
            raise LoaderError(
                f"Malformed CSV at line {reader.reader.line_num} in {source}: {exc}"
            ) from exc
        except (OSError, UnicodeError) as exc:
            raise LoaderError(f"Could not read {source}: {exc}") from exc
        return documents


class JSONLLoader(DocumentLoader):
    """Load one JSON object per line as a document.

    Args:
        content_key: String field to use as content, or a list of string fields
            to join in order with ``separator``.
        metadata_keys: Fields to copy into metadata, retaining their JSON
            types. Other fields are omitted. ``row_index`` is reserved for the
            zero-based line index; selecting ``source`` overrides the file path.
        id_key: String or integer field to use as the document ID (as a string).
            Otherwise, IDs use the source path hash and zero-based line index.
        separator: Text between multiple content fields (a newline by default).
        encoding: File encoding.

    Empty files produce no documents. Blank lines, non-object records, and
    malformed JSON raise ``LoaderError`` with the one-based line number.
    """

    def __init__(
        self,
        content_key: str | list[str] = "text",
        metadata_keys: list[str] | None = None,
        id_key: str | None = None,
        *,
        separator: str = "\n",
        encoding: str = "utf-8",
    ) -> None:
        if not content_key:
            raise ValueError("content_key must not be empty.")
        if metadata_keys and "row_index" in metadata_keys:
            raise ValueError("row_index is reserved for the line index.")
        self.content_keys = [content_key] if isinstance(content_key, str) else list(content_key)
        self.metadata_keys = list(metadata_keys or [])
        self.id_key = id_key
        self.separator = separator
        self.encoding = encoding

    def load(self, source: str) -> list[Document]:
        """Read JSON objects, raising ``LoaderError`` for file or record errors."""
        documents: list[Document] = []
        try:
            with Path(source).open(encoding=self.encoding) as stream:
                for row_index, line in enumerate(stream):
                    location = f"line {row_index + 1}"
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LoaderError(
                            f"Malformed JSON at {location} in {source}: {exc}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise LoaderError(f"Expected a JSON object at {location} in {source}.")
                    documents.append(
                        _document_from_record(
                            record,
                            source,
                            row_index,
                            location,
                            self.content_keys,
                            self.metadata_keys,
                            self.id_key,
                            self.separator,
                        )
                    )
        except (OSError, UnicodeError) as exc:
            raise LoaderError(f"Could not read {source}: {exc}") from exc
        return documents
