"""File-based tests for CSV and JSON Lines document loaders."""

import json

import pytest

from ragframework.base import DocumentLoader
from ragframework.document import CSVLoader, JSONLLoader
from ragframework.document.loaders import _make_id
from ragframework.exceptions import LoaderError


@pytest.fixture(params=["csv", "jsonl"])
def loader(request):
    return CSVLoader(["text"]) if request.param == "csv" else JSONLLoader()


@pytest.mark.parametrize("line_ending", ["\n", "\r\n"])
def test_csv_selected_fields_and_quoted_multiline_content(tmp_path, line_ending):
    path = tmp_path / "records.csv"
    contents = (
        "id,title,body,url,private\n"
        'a,"Hello, world","First line\nSecond line",https://example.com,a secret\n'
        "b,Another,Record,https://example.org,another secret\n"
    )
    path.write_bytes(contents.replace("\n", line_ending).encode("utf-8"))
    loader = CSVLoader(["title", "body"], metadata_columns=["url"], id_column="id")

    docs = loader.load(str(path))

    assert isinstance(loader, DocumentLoader)
    assert [doc.id for doc in docs] == ["a", "b"]
    assert [doc.content for doc in docs] == [
        f"Hello, world\nFirst line{line_ending}Second line",
        "Another\nRecord",
    ]
    assert docs[0].metadata == {"source": str(path), "row_index": 0, "url": "https://example.com"}
    assert docs[1].metadata == {"source": str(path), "row_index": 1, "url": "https://example.org"}


def test_csv_custom_delimiter_encoding_and_separator(tmp_path):
    path = tmp_path / "records.csv"
    path.write_text("title;body\ncafé;crème\n", encoding="latin-1")
    docs = CSVLoader(["title", "body"], delimiter=";", separator=" | ", encoding="latin-1").load(
        str(path)
    )
    assert docs[0].content == "café | crème"


def test_jsonl_selected_metadata_preserves_types_and_explicit_ids(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [
        {"id": 0, "text": "First", "source": "export", "tags": ["faq"], "extra": "private"},
        {"id": "second", "text": "Second", "source": "web", "tags": None},
    ]
    path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    loader = JSONLLoader(metadata_keys=["source", "tags"], id_key="id")

    docs = loader.load(str(path))

    assert isinstance(loader, DocumentLoader)
    assert [doc.id for doc in docs] == ["0", "second"]
    assert [doc.content for doc in docs] == ["First", "Second"]
    assert docs[0].metadata == {"source": "export", "tags": ["faq"], "row_index": 0}
    assert docs[1].metadata == {"source": "web", "tags": None, "row_index": 1}


@pytest.mark.parametrize("separator", ["\n", " | "])
def test_jsonl_multiple_content_keys_and_encoding(tmp_path, separator):
    path = tmp_path / "records.jsonl"
    path.write_text('{"title":"café","body":"crème"}\n', encoding="latin-1")
    docs = JSONLLoader(content_key=["title", "body"], separator=separator, encoding="latin-1").load(
        str(path)
    )
    assert docs[0].content == separator.join(["café", "crème"])


def test_default_ids_and_metadata_are_stable(tmp_path, loader):
    path = tmp_path / "records"
    contents = (
        "text,unused\nFirst,secret\nSecond,private\n"
        if isinstance(loader, CSVLoader)
        else '{"text":"First","unused":"secret"}\n{"text":"Second","unused":"private"}\n'
    )
    path.write_text(contents, encoding="utf-8")

    docs = loader.load(str(path))

    assert [doc.id for doc in docs] == [f"{_make_id(str(path))}:0", f"{_make_id(str(path))}:1"]
    assert [doc.content for doc in docs] == ["First", "Second"]
    assert [doc.metadata for doc in docs] == [
        {"source": str(path), "row_index": 0},
        {"source": str(path), "row_index": 1},
    ]
    assert loader.load(str(path)) == docs


def test_empty_file(tmp_path, loader):
    path = tmp_path / "empty"
    path.write_text("", encoding="utf-8")
    assert loader.load(str(path)) == []


def test_csv_header_only(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("text\n", encoding="utf-8")
    assert CSVLoader(["text"]).load(str(path)) == []


@pytest.mark.parametrize(
    "options",
    [
        {"content_columns": ["missing"]},
        {"content_columns": ["text"], "metadata_columns": ["missing"]},
        {"content_columns": ["text"], "id_column": "missing"},
    ],
)
def test_csv_missing_selected_column_reports_row(tmp_path, options):
    path = tmp_path / "missing.csv"
    path.write_text("text\nHello\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="Missing field 'missing' at row 1"):
        CSVLoader(**options).load(str(path))


def test_csv_short_row_reports_missing_value(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("title,body\nFirst,Complete\nSecond\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="Missing field 'body' at row 2"):
        CSVLoader(["title", "body"]).load(str(path))


def test_csv_explicit_empty_field_is_valid(tmp_path):
    path = tmp_path / "blank.csv"
    path.write_text("title,body\nTitle,\n", encoding="utf-8")
    assert CSVLoader(["title", "body"]).load(str(path))[0].content == "Title\n"


def test_csv_extra_fields_raise(tmp_path):
    path = tmp_path / "extra.csv"
    path.write_text("text\nHello,extra\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="Extra CSV fields at row 1"):
        CSVLoader(["text"]).load(str(path))


def test_csv_malformed_quotes_raise_loader_error(tmp_path):
    path = tmp_path / "malformed.csv"
    path.write_text('text\n"unterminated\n', encoding="utf-8")
    with pytest.raises(LoaderError, match="Malformed CSV at line 2") as exc:
        CSVLoader(["text"]).load(str(path))
    assert exc.value.__cause__ is not None


@pytest.mark.parametrize(
    "options",
    [
        {"content_key": "missing"},
        {"metadata_keys": ["missing"]},
        {"id_key": "missing"},
    ],
)
def test_jsonl_missing_selected_key_reports_line(tmp_path, options):
    path = tmp_path / "missing.jsonl"
    path.write_text('{"text":"First","missing":"present"}\n{"text":"Second"}\n', encoding="utf-8")
    with pytest.raises(LoaderError, match="Missing field 'missing' at line 2"):
        JSONLLoader(**options).load(str(path))


@pytest.mark.parametrize("line", ["{broken}", "", "   "])
def test_jsonl_malformed_or_blank_line_reports_line_number(tmp_path, line):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"text":"Valid"}\n' + line + "\n", encoding="utf-8")
    with pytest.raises(LoaderError, match="Malformed JSON at line 2") as exc:
        JSONLLoader().load(str(path))
    assert isinstance(exc.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize("value", [None, [], 42, "text"])
def test_jsonl_requires_object_records(tmp_path, value):
    path = tmp_path / "nonobject.jsonl"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(LoaderError, match="Expected a JSON object at line 1"):
        JSONLLoader().load(str(path))


@pytest.mark.parametrize("value", [None, [], 42])
def test_jsonl_requires_string_content(tmp_path, value):
    path = tmp_path / "content.jsonl"
    path.write_text(json.dumps({"text": value}), encoding="utf-8")
    with pytest.raises(LoaderError, match="Content field 'text' at line 1 .* must be a string"):
        JSONLLoader().load(str(path))


@pytest.mark.parametrize("value", [None, [], {}, True])
def test_jsonl_rejects_invalid_ids(tmp_path, value):
    path = tmp_path / "id.jsonl"
    path.write_text(json.dumps({"text": "Hello", "id": value}), encoding="utf-8")
    with pytest.raises(LoaderError, match="ID field 'id' at line 1 .* must be a string or integer"):
        JSONLLoader(id_key="id").load(str(path))


def test_file_errors_are_wrapped(tmp_path, loader):
    with pytest.raises(LoaderError, match="Could not read") as exc:
        loader.load(str(tmp_path / "missing"))
    assert isinstance(exc.value.__cause__, OSError)
    with pytest.raises(LoaderError, match="Could not read"):
        loader.load(str(tmp_path))


def test_decode_errors_are_wrapped(tmp_path, loader):
    path = tmp_path / "invalid-utf8"
    path.write_bytes(b"\xff")
    with pytest.raises(LoaderError, match="Could not read") as exc:
        loader.load(str(path))
    assert isinstance(exc.value.__cause__, UnicodeDecodeError)


@pytest.mark.parametrize("delimiter", ["", "::", "\n", "\r"])
def test_csv_rejects_invalid_delimiter(delimiter):
    with pytest.raises(ValueError, match="delimiter"):
        CSVLoader(["text"], delimiter=delimiter)


def test_empty_content_selection_and_reserved_metadata_are_rejected():
    with pytest.raises(ValueError, match="content_columns"):
        CSVLoader([])
    with pytest.raises(ValueError, match="content_key"):
        JSONLLoader(content_key=[])
    with pytest.raises(ValueError, match="row_index"):
        CSVLoader(["text"], metadata_columns=["row_index"])
    with pytest.raises(ValueError, match="row_index"):
        JSONLLoader(metadata_keys=["row_index"])
