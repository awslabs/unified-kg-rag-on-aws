# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ParserFactory format support + optional-dependency gating.

Markdown and HTML parse in the core install (the data-plane image once could
not read a .md at all: its loader needed the optional `unstructured`); the
unstructured extra, when installed, is the richer parser of both. .docx
needs the docx extra and is registered only with it.
"""

from __future__ import annotations

# Imported here, not first inside the 10 s test below: pypdf's first import
# takes seconds on a loaded runner (one pytest-xdist worker imports it there).
import random
import sys

import pypdf  # noqa: F401
import pytest

from tests.fixtures.extras import DOCX_INSTALLED
from unified_kg_rag.adapters.ingestion.file_loaders import (
    DocxFileLoader,
    HtmlFileLoader,
    TextFileLoader,
    UnstructuredPartitionLoader,
)
from unified_kg_rag.adapters.ingestion.parser import (
    _UNSTRUCTURED_AVAILABLE,
    MISSING_EXTRA_FORMATS,
    FileParser,
    ParserFactory,
)
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import DataProcessingError

pytestmark = pytest.mark.unit


def test_core_formats_always_supported() -> None:
    exts = set(ParserFactory.get_supported_extensions())
    # No-heavy-dep loaders must always be available.
    assert {".txt", ".csv", ".json", ".pdf"} <= exts


def test_markdown_and_html_need_no_extra() -> None:
    exts = set(ParserFactory.get_supported_extensions())
    assert {".md", ".markdown", ".html", ".htm"} <= exts
    configs = ParserFactory._loader_configs
    # unstructured, when installed, is the richer parser and takes them over.
    expected = (
        (UnstructuredPartitionLoader, UnstructuredPartitionLoader)
        if _UNSTRUCTURED_AVAILABLE
        else (TextFileLoader, HtmlFileLoader)
    )
    assert (configs[".md"][0], configs[".html"][0]) == expected


def test_docx_follows_its_extra() -> None:
    exts = set(ParserFactory.get_supported_extensions())
    if DOCX_INSTALLED:
        assert ".docx" in exts and ".docx" not in MISSING_EXTRA_FORMATS
        return
    assert ".docx" not in exts
    with pytest.raises(DataProcessingError, match="extra docx"):
        ParserFactory.create_parser("report.docx", Config())


def test_a_docx_without_its_extra_fails_naming_it(tmp_path, monkeypatch) -> None:
    # The loader itself, registered by hand on an install without python-docx.
    monkeypatch.setitem(sys.modules, "docx", None)
    path = tmp_path / "report.docx"
    path.write_bytes(b"PK")
    parser = FileParser(Config(), DocxFileLoader, {}, "Word")
    with pytest.raises(DataProcessingError, match="uv sync --inexact --extra docx"):
        parser.parse_file(path)


@pytest.mark.skipif(_UNSTRUCTURED_AVAILABLE, reason="unstructured reads .md")
def test_core_markdown_is_read_as_written(tmp_path) -> None:
    path = tmp_path / "terms.md"
    path.write_text("# 공급 조건\n\n**공급자**는 납품한다.\n", encoding="utf-8")
    doc = ParserFactory.create_parser(path, Config()).parse_file(path)
    assert doc.page_content == "# 공급 조건\n\n**공급자**는 납품한다.\n"


@pytest.mark.skipif(_UNSTRUCTURED_AVAILABLE, reason="unstructured reads .html")
def test_core_html_retries_a_non_utf8_file(tmp_path) -> None:
    path = tmp_path / "terms.html"
    path.write_bytes("<p>공급자는 구매자에게 물품을 납품한다.</p>".encode("cp949"))
    doc = ParserFactory.create_parser(path, Config()).parse_file(path)
    assert doc.page_content == "공급자는 구매자에게 물품을 납품한다."


def test_a_binary_file_with_a_text_suffix_fails_to_parse(tmp_path) -> None:
    # Regression (rv24 quality 3): random bytes detect as no encoding, and
    # the latin-1 fallback indexed them as text units of garbage.
    path = tmp_path / "binary.txt"
    path.write_bytes(random.Random(7).randbytes(3072))
    with pytest.raises(DataProcessingError, match="not text in any detected"):
        ParserFactory.create_parser(path, Config()).parse_file(path)


def test_the_configured_encoding_reads_a_non_utf8_file(tmp_path) -> None:
    # charset-normalizer reads this French cp1252 file as cp1250 ("ŕ
    # Genčve"); the configured encoding is read instead of a detection.
    text = "Livraison à Genève, réception déjà confirmée.\n"
    path = tmp_path / "terms.txt"
    path.write_bytes(text.encode("cp1252"))
    config = Config()
    config.processing.document_parsing.encoding = "cp1252"
    assert ParserFactory.create_parser(path, config).parse_file(path).page_content == (
        text
    )
    # A UTF-8 file is read as UTF-8 whatever is configured.
    path.write_text("Livraison à Genève.\n", encoding="utf-8")
    doc = ParserFactory.create_parser(path, config).parse_file(path)
    assert doc.page_content == "Livraison à Genève.\n"


def test_an_unknown_encoding_fails_the_config_load() -> None:
    with pytest.raises(ValueError, match="unknown encoding 'utf-99'"):
        Config.model_validate(
            {"processing": {"document_parsing": {"encoding": "utf-99"}}}
        )
    assert (
        Config.model_validate(
            {"processing": {"document_parsing": {"encoding": "Latin1"}}}
        ).processing.document_parsing.encoding
        == "iso8859-1"
    )


def test_core_parser_constructs() -> None:
    # A supported core format yields a parser without touching AWS.
    parser = ParserFactory.create_parser("doc.txt", Config())
    assert parser is not None


@pytest.mark.timeout(10)
def test_truncated_pdf_xref_fails_fast(tmp_path) -> None:
    # Upstream fixture of GHSA-5849-834g-hvw5: an xref section that hits EOF
    # before its trailer. pypdf before 6.21.4 loops on it forever; the parser
    # must reject it as a parse error instead.
    pdf = tmp_path / "truncated.pdf"
    pdf.write_bytes(b"%PDF-1.4\nstartxref\n28\n%%EOF\nxref\n1 0\naaa")
    parser = ParserFactory.create_parser(pdf, Config())
    with pytest.raises(DataProcessingError, match="Xref parsing stuck"):
        parser.parse_file(pdf)


# --- register_loader: custom-format extension seam --------------------------

from langchain_core.document_loaders import BaseLoader  # noqa: E402


@pytest.fixture
def _restore_loader_configs():
    # register_loader mutates class-level _loader_configs; snapshot + restore so
    # registration tests don't leak into others.
    saved = dict(ParserFactory._loader_configs)
    yield
    ParserFactory._loader_configs = saved


class _DummyLoader(BaseLoader):
    def __init__(self, file_path, **kwargs):  # noqa: ANN001, ANN003
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        return []


def test_register_loader_adds_extension(_restore_loader_configs) -> None:
    ParserFactory.register_loader(".xyz", _DummyLoader, file_type_name="XYZ")
    assert ".xyz" in ParserFactory.get_supported_extensions()
    parser = ParserFactory.create_parser("data.xyz", Config())
    assert parser.loader_class is _DummyLoader
    assert parser.file_type_name == "XYZ"


def test_register_loader_default_type_name(_restore_loader_configs) -> None:
    ParserFactory.register_loader(".abc", _DummyLoader)
    parser = ParserFactory.create_parser("f.abc", Config())
    assert parser.file_type_name == "ABC"  # derived from extension


def test_register_loader_passes_kwargs(_restore_loader_configs) -> None:
    ParserFactory.register_loader(".kv", _DummyLoader, loader_kwargs={"mode": "x"})
    parser = ParserFactory.create_parser("f.kv", Config())
    assert parser.loader_kwargs == {"mode": "x"}


def test_register_loader_can_override_builtin(_restore_loader_configs) -> None:
    ParserFactory.register_loader(".txt", _DummyLoader)
    parser = ParserFactory.create_parser("f.txt", Config())
    assert parser.loader_class is _DummyLoader


def test_register_loader_rejects_missing_dot(_restore_loader_configs) -> None:
    with pytest.raises(ValueError, match="Cannot register a file loader as .xyz."):
        ParserFactory.register_loader("xyz", _DummyLoader)


def test_register_loader_rejects_non_loader(_restore_loader_configs) -> None:
    with pytest.raises(TypeError, match="BaseLoader subclass"):
        ParserFactory.register_loader(".xyz", dict)  # type: ignore[arg-type]
