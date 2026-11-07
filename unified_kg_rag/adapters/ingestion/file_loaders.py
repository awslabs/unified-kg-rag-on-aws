# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-file loaders for the parser, on langchain-core only.

They replace the langchain-community ``TextLoader``, ``CSVLoader``,
``PyPDFLoader`` and ``UnstructuredMarkdownLoader``/``UnstructuredHTMLLoader``
(that package is in sunset) and produce the same ``page_content`` and
metadata, so an already-indexed corpus keeps its content hashes and
``document_id``s: an incremental run after the switch finds no changed
document. ``tests/unit/test_file_loaders_golden.py`` pins the parity.

* :class:`TextFileLoader` reads in text mode, so universal newlines turn CRLF
  and bare CR into ``\\n`` exactly as before.
* :class:`CsvFileLoader` emits one document per row, ``key: value`` lines.
* :class:`PdfFileLoader` emits one document per page (``pypdf`` plain-mode
  text, stripped) with the PDF's info dictionary as metadata.
* :class:`HtmlFileLoader` turns HTML into text with the standard library's
  parser: one paragraph per block element, scripts and styles dropped.
* :class:`DocxFileLoader` reads a Word document's paragraphs and tables in
  order (``python-docx``, the optional ``docx`` extra).
* :class:`UnstructuredPartitionLoader` joins the elements of an
  ``unstructured`` partition (the optional ``unstructured`` extra), the richer
  Markdown and HTML parser: when it is installed it reads those formats in
  place of :class:`TextFileLoader` (Markdown as written) and
  :class:`HtmlFileLoader`.

Encoding detection is not done here: ``FileParser`` retries a
``UnicodeDecodeError`` with the encoding charset-normalizer detects.
"""

from __future__ import annotations

import csv
import importlib
import importlib.util
import re
import threading
from collections.abc import Iterator
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document as LangChainDocument

from unified_kg_rag.shared.exceptions import DataProcessingError


class TextFileLoader(BaseLoader):
    """Load a text file as one document.

    ``encoding=None`` reads with the locale's preferred encoding, as
    ``open()`` does (and ``TextLoader`` did).
    """

    def __init__(self, file_path: str | Path, encoding: str | None = None) -> None:
        self.file_path = file_path
        self.encoding = encoding

    def lazy_load(self) -> Iterator[LangChainDocument]:
        with open(self.file_path, encoding=self.encoding) as file:
            text = file.read()
        yield LangChainDocument(
            page_content=text, metadata={"source": str(self.file_path)}
        )


class CsvFileLoader(BaseLoader):
    """Load a CSV file as one document per row.

    A row becomes one ``column: value`` line per column (values stripped).
    A row longer than the header lists its extra values under ``None``
    (comma-joined), and a short row shows ``None`` for its missing values,
    as ``csv.DictReader`` reports them. ``csv_args`` are passed to
    ``csv.DictReader`` (``delimiter``, ``quotechar``, ``fieldnames``...).
    """

    def __init__(
        self,
        file_path: str | Path,
        encoding: str | None = None,
        csv_args: dict[str, Any] | None = None,
    ) -> None:
        self.file_path = file_path
        self.encoding = encoding
        self.csv_args = csv_args or {}

    def lazy_load(self) -> Iterator[LangChainDocument]:
        # newline="" lets the csv module see the quoted newlines inside a
        # field; it keeps them as written.
        with open(self.file_path, newline="", encoding=self.encoding) as file:
            for row_number, row in enumerate(csv.DictReader(file, **self.csv_args)):
                content = "\n".join(
                    f"{_strip(key)}: {_cell(value)}" for key, value in row.items()
                )
                yield LangChainDocument(
                    page_content=content,
                    metadata={"source": str(self.file_path), "row": row_number},
                )


def _strip(key: str | None) -> str | None:
    return key.strip() if key is not None else key


def _cell(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return ",".join(map(str.strip, value))
    return value


class PdfFileLoader(BaseLoader):
    """Load a PDF as one document per page with ``pypdf``.

    The page text is ``extract_text(extraction_mode="plain")``, stripped.
    Every page's metadata holds the info dictionary (keys lower-cased without
    the leading ``/``, dates as ISO 8601) plus ``source``, ``total_pages``,
    ``page`` (0-based) and ``page_label``. A local file only: unlike
    ``PyPDFLoader``, a URL is never downloaded.
    """

    def __init__(self, file_path: str | Path, password: str | None = None) -> None:
        self.file_path = file_path
        self.password = password

    def lazy_load(self) -> Iterator[LangChainDocument]:
        import pypdf

        with open(self.file_path, "rb") as file:
            reader = pypdf.PdfReader(file, password=self.password)
            metadata = _pdf_metadata(
                {"producer": "PyPDF", "creator": "PyPDF", "creationdate": ""}
                | dict(reader.metadata or {})
                | {"source": str(self.file_path), "total_pages": len(reader.pages)}
            )
            # Built for every page on each read: once per file, not per page.
            page_labels = reader.page_labels
            for page_number, page in enumerate(reader.pages):
                yield LangChainDocument(
                    page_content=page.extract_text(extraction_mode="plain").strip(),
                    metadata=metadata
                    | {"page": page_number, "page_label": page_labels[page_number]},
                )


_PDF_DATE_KEYS = frozenset({"creationdate", "moddate"})


def _pdf_metadata(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a PDF info dictionary the way ``PyPDFLoader`` did."""
    metadata: dict[str, Any] = {}
    for key, value in raw.items():
        if type(value) not in (str, int):
            value = str(value)
        key = key.removeprefix("/").lower()
        if key in _PDF_DATE_KEYS:
            try:
                metadata[key] = datetime.strptime(
                    value.replace("'", ""), "D:%Y%m%d%H%M%S%z"
                ).isoformat("T")
            except ValueError:
                metadata[key] = value
        elif key == "page_count":
            metadata["total_pages"] = metadata[key] = value
        elif key == "file_path":
            metadata["source"] = metadata[key] = value
        elif isinstance(value, str):
            metadata[key] = value.strip()
        else:
            metadata[key] = value
    return metadata


# Elements whose text is not the page's: dropped with everything inside.
# Not <head>, often left unclosed: its only text is the <title>, read apart.
_HTML_SKIPPED = frozenset(
    {"script", "style", "noscript", "template", "svg", "math", "iframe"}
)
# Elements that start and end a paragraph of the text.
_HTML_BLOCKS = frozenset(
    {
        "address", "article", "aside", "blockquote", "caption", "dd", "details",
        "dialog", "div", "dl", "dt", "fieldset", "figcaption", "figure",
        "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr",
        "legend", "li", "main", "nav", "ol", "p", "pre", "section", "summary",
        "table", "tbody", "tfoot", "thead", "tr", "ul",
    }
)  # fmt: skip
_HTML_CELLS = frozenset({"td", "th"})
_WHITESPACE = re.compile(r"\s+")


class _HtmlText(HTMLParser):
    """The text of an HTML document as paragraphs (see HtmlFileLoader)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._parts: list[str] = []
        self._skip_depth = 0
        self._pre_depth = 0
        self._title: list[str] = []
        self._in_title = False
        self._row_has_cell = False

    def _flush(self) -> None:
        text = "".join(self._parts)
        lines = [line.strip() for line in text.split("\n")]
        block = "\n".join(line for line in lines if line)
        if block:
            self.blocks.append(block)
        self._parts = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag == "title":
            self._in_title = True
            return
        if tag in _HTML_SKIPPED:
            self._skip_depth += 1
        if self._skip_depth:
            return
        if tag in _HTML_BLOCKS:
            self._flush()
            self._row_has_cell = False
            if tag == "li":
                self._parts.append("- ")
        elif tag in _HTML_CELLS:
            if self._row_has_cell:
                self._parts.append(" | ")
            self._row_has_cell = True
        elif tag == "br":
            self._parts.append("\n")
        if tag == "pre":
            self._pre_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if tag in _HTML_SKIPPED:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "pre":
            self._pre_depth = max(0, self._pre_depth - 1)
        if tag in _HTML_BLOCKS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title.append(data)
            return
        self._parts.append(data if self._pre_depth else _WHITESPACE.sub(" ", data))

    def text(self) -> str:
        self._flush()
        title = _WHITESPACE.sub(" ", "".join(self._title)).strip()
        return "\n\n".join([title, *self.blocks] if title else self.blocks)


def html_to_text(html: str) -> str:
    """``html`` as text: the title, then one paragraph per block element
    (paragraphs, headings, list items as ``- item``, table rows with their
    cells joined by `` | ``), whitespace collapsed (``<pre>`` keeps its line
    breaks), and no script, style or other non-content element."""
    parser = _HtmlText()
    parser.feed(html)
    parser.close()
    return parser.text()


class HtmlFileLoader(BaseLoader):
    """Load an HTML file as one document of its text (:func:`html_to_text`),
    with the standard library alone. ``encoding=None`` reads as ``open()``
    does; the parser retries a decode error with the detected encoding."""

    def __init__(self, file_path: str | Path, encoding: str | None = None) -> None:
        self.file_path = file_path
        self.encoding = encoding

    def lazy_load(self) -> Iterator[LangChainDocument]:
        with open(self.file_path, encoding=self.encoding) as file:
            text = html_to_text(file.read())
        yield LangChainDocument(
            page_content=text, metadata={"source": str(self.file_path)}
        )


DOCX_INSTALL_HINT = (
    "install the optional 'docx' extra: `uv sync --extra docx` or "
    "`pip install -e '.[docx]'` in a source checkout (the docker/Dockerfile "
    'image: build with `--build-arg UV_EXTRAS="--extra docx"`)'
)


class DocxFileLoader(BaseLoader):
    """Load a Word (.docx) document as one document: its body's paragraphs
    and tables in order, a table row's cells joined by `` | ``; headers,
    footers, comments and images are not read. Needs the optional ``docx``
    extra (``python-docx``, imported when a file is loaded)."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = file_path

    def lazy_load(self) -> Iterator[LangChainDocument]:
        try:
            docx = importlib.import_module("docx")
        except ImportError as exc:
            raise DataProcessingError(
                f"reading a .docx file needs python-docx; {DOCX_INSTALL_HINT}"
            ) from exc
        document = docx.Document(str(self.file_path))
        blocks: list[str] = []
        for item in document.iter_inner_content():
            if hasattr(item, "rows"):
                for row in item.rows:
                    # A merged cell is listed once per grid column it spans.
                    cells = {id(cell._tc): cell.text.strip() for cell in row.cells}
                    line = " | ".join(text for text in cells.values() if text)
                    if line:
                        blocks.append(line)
            elif text := item.text.strip():
                blocks.append(text)
        metadata: dict[str, Any] = {"source": str(self.file_path)}
        if title := (document.core_properties.title or "").strip():
            metadata["title"] = title
        yield LangChainDocument(page_content="\n\n".join(blocks), metadata=metadata)


PartitionFormat = Literal["html", "md"]

# The spaCy model unstructured's partitioners tokenize with. When it is not
# installed, unstructured downloads it from GitHub on first use, waiting up to
# 120 s, and tries again for every file: with no network (a private VPC) each
# Markdown or HTML file waited that long and failed alike. The image bakes it
# in (docker/Dockerfile); elsewhere the first failure is kept for the process
# and every later file fails at once with it.
SPACY_MODEL = "en_core_web_sm"
SPACY_MODEL_INSTALL = (
    'python -c "from unstructured.nlp.tokenize import sent_tokenize; '
    "sent_tokenize('Ready.')\" (once, with network access; docker/Dockerfile "
    "does it at build time with the unstructured extra)"
)
_spacy_lock = threading.Lock()
# None until checked, then "" when the model is installed, else why it is not.
_spacy_problem: str | None = None


def require_spacy_model() -> None:
    """Make sure the spaCy model unstructured needs is installed, downloading
    it once (unstructured's own download) when it is not.

    Raises:
        DataProcessingError: The model is not installed and could not be
            installed; raised at once for every later call of the process.
    """
    global _spacy_problem
    with _spacy_lock:
        if _spacy_problem is None:
            _spacy_problem = _spacy_model_problem()
        problem = _spacy_problem
    if problem:
        raise DataProcessingError(problem)


def _spacy_model_problem() -> str:
    """Why the spaCy model is not installed and could not be, or an empty
    string once it is (downloaded now when it was not)."""
    if importlib.util.find_spec(SPACY_MODEL) is not None:
        return ""
    try:
        tokenize = importlib.import_module("unstructured.nlp.tokenize")
        tokenize.sent_tokenize("Ready.")
    except (OSError, RuntimeError) as exc:
        return (
            f"the spaCy model {SPACY_MODEL} that unstructured parses Markdown "
            f"and HTML with is not installed and could not be downloaded "
            f"({exc}); install it with {SPACY_MODEL_INSTALL}"
        )
    return ""


class UnstructuredPartitionLoader(BaseLoader):
    """Load a Markdown or HTML file as one document with ``unstructured``.

    The text is the partition's elements joined by blank lines. Needs the
    optional ``unstructured`` extra (imported when a file is loaded).
    """

    def __init__(
        self, file_path: str | Path, partition: PartitionFormat = "md"
    ) -> None:
        if partition not in ("html", "md"):
            raise ValueError(f"partition must be 'html' or 'md', got '{partition}'")
        self.file_path = file_path
        self.partition = partition

    def lazy_load(self) -> Iterator[LangChainDocument]:
        # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import - html or md, validated
        module = importlib.import_module(f"unstructured.partition.{self.partition}")
        require_spacy_model()
        partition = getattr(module, f"partition_{self.partition}")
        elements = partition(filename=str(self.file_path))
        yield LangChainDocument(
            page_content="\n\n".join(str(element) for element in elements),
            metadata={"source": str(self.file_path)},
        )
