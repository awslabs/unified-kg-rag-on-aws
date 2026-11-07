# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The loaders that read HTML (standard library) and Word (.docx, the docx
extra) source files, and how ingestion reports a format whose extra is
missing."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from tests.fixtures.extras import requires_docx
from unified_kg_rag.adapters.ingestion import parser as parser_module
from unified_kg_rag.adapters.ingestion.file_loaders import (
    DOCX_INSTALL_HINT,
    DocxFileLoader,
    HtmlFileLoader,
    html_to_text,
)
from unified_kg_rag.application.ingestion.pipeline_stages import (
    DocumentParsingStage,
)
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.unit


def test_html_text_keeps_the_content_and_drops_the_rest() -> None:
    html = (
        "<html><head><title>Supply &amp; terms</title>"
        "<style>p { color: red }</style><script>var x = '<p>no</p>';</script>"
        "</head><body><nav>Home</nav><h1>Supply   terms</h1>"
        "<p>The Vendor\n   ships parts.<br>Buyer pays.</p>"
        "<ul><li>First</li><li>Second</li></ul>"
        "<table><tr><th>Term</th><th>Value</th></tr>"
        "<tr><td>Days</td><td>30</td></tr></table>"
        "<noscript>Enable JavaScript</noscript></body></html>"
    )

    assert html_to_text(html) == (
        "Supply & terms\n\nHome\n\nSupply terms\n\n"
        "The Vendor ships parts.\nBuyer pays.\n\n- First\n\n- Second\n\n"
        "Term | Value\n\nDays | 30"
    )


@pytest.mark.parametrize(
    ("html", "text"),
    [
        (
            "<p>공급자는 구매자에게&nbsp;물품을 납품한다.</p>",
            "공급자는 구매자에게 물품을 납품한다.",
        ),
        ("<div>東邦商事は<b>倉庫</b>を借りる。</div>", "東邦商事は倉庫を借りる。"),
        (
            "<p>供应商向买方交付货物。</p><p>第二段。</p>",
            "供应商向买方交付货物。\n\n第二段。",
        ),
    ],
    ids=["ko", "ja", "zh"],
)
def test_html_text_of_dense_scripts_is_not_split(html: str, text: str) -> None:
    # Inline elements join their text as written: no space is added between
    # Han, Kana or Hangul runs.
    assert html_to_text(html).replace("\xa0", " ") == text


def test_an_unclosed_head_does_not_hide_the_body() -> None:
    # Browsers end <head> at <body>; the parser reads the body all the same.
    assert html_to_text("<head><title>T</title><body><p>Body text") == "T\n\nBody text"


def test_html_file_loader_reads_one_document(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text("<h2>Clause</h2><p>Notice: 60 days.</p>", encoding="utf-8")

    (doc,) = HtmlFileLoader(path).load()

    assert doc.page_content == "Clause\n\nNotice: 60 days."
    assert doc.metadata == {"source": str(path)}


@requires_docx
def test_docx_paragraphs_and_tables_are_read_in_order(tmp_path: Path) -> None:
    import docx

    document = docx.Document()
    document.core_properties.title = "공급 계약"
    document.add_heading("Supply terms", level=1)
    document.add_paragraph("공급자는 구매자에게 물품을 납품한다.")
    table = document.add_table(rows=2, cols=3)
    table.cell(0, 0).text = "Term"
    table.cell(0, 1).merge(table.cell(0, 2)).text = "Value"
    for column, value in enumerate(("Days", "30", "30")):
        table.cell(1, column).text = value
    document.add_paragraph("")
    document.add_paragraph("東邦商事は倉庫を借りる。")
    path = tmp_path / "terms.docx"
    document.save(str(path))

    (doc,) = DocxFileLoader(path).load()

    # A merged cell is read once; equal values of separate cells are kept.
    assert doc.page_content == (
        "Supply terms\n\n공급자는 구매자에게 물품을 납품한다.\n\n"
        "Term | Value\n\nDays | 30 | 30\n\n東邦商事は倉庫を借りる。"
    )
    assert doc.metadata == {"source": str(path), "title": "공급 계약"}
    parsed = parser_module.ParserFactory.create_parser(path, Config()).parse_file(path)
    assert parsed.page_content == doc.page_content


def test_a_skipped_format_names_the_extra_that_reads_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # An install without the docx extra: .docx is not registered.
    configs = dict(parser_module.ParserFactory._loader_configs)
    configs.pop(".docx", None)
    monkeypatch.setattr(parser_module.ParserFactory, "_loader_configs", configs)
    monkeypatch.setitem(parser_module.MISSING_EXTRA_FORMATS, ".docx", DOCX_INSTALL_HINT)
    (tmp_path / "notes.md").write_text("# Notes\n\nVendor ships.\n", encoding="utf-8")
    (tmp_path / "report.docx").write_bytes(b"PK")
    (tmp_path / "data.xyz").write_text("x", encoding="utf-8")
    stage = DocumentParsingStage(Config(), source_directory=tmp_path)

    with caplog.at_level(logging.WARNING):
        files, skipped = stage._discover_source_files()

    assert [f.name for f in files] == ["notes.md"]
    assert sorted(Path(s).name for s in skipped) == ["data.xyz", "report.docx"]
    (warning,) = [
        r.getMessage() for r in caplog.records if "Skipping" in r.getMessage()
    ]
    assert "for .docx, install the optional 'docx' extra" in warning
    assert "for .xyz, convert them to a supported format" in warning
