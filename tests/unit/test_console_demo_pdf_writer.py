# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console demo's PDF writer (``console/demo/pdf_writer.py``): its pages
parse with pypdf (the core install's PDF loader) one line per laid-out line,
a table row as one line of its cells, and the same pages give the same bytes."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.console_demo import load
from unified_kg_rag.adapters.ingestion.file_loaders import PdfFileLoader

pytestmark = pytest.mark.unit


def _pages() -> list[list[object]]:
    pdf = load("pdf_writer")
    return [
        [
            pdf.Heading("Supply Terms", 16),
            pdf.Paragraph("Vendor ships parts to Buyer (weekly).\nBackslash \\ kept."),
            pdf.Table((("Term", "Figure"), ("Penalty", "$100 per day"))),
        ],
        [pdf.Heading("Annex"), pdf.Paragraph("Prices in EUR: 5 € each – net.")],
    ]


def test_each_page_extracts_as_laid_out(tmp_path: Path) -> None:
    pdf = load("pdf_writer")
    path = tmp_path / "terms.pdf"
    path.write_bytes(pdf.render(_pages(), title="Supply Terms"))
    pages = PdfFileLoader(path).load()
    assert [p.page_content for p in pages] == [
        "Supply Terms\nVendor ships parts to Buyer (weekly).\nBackslash \\ kept.\n"
        "Term Figure\nPenalty $100 per day",
        "Annex\nPrices in EUR: 5 € each – net.",
    ]
    assert [p.metadata["page"] for p in pages] == [0, 1]
    assert pages[0].metadata["title"] == "Supply Terms"
    assert pages[0].metadata["total_pages"] == 2


def test_the_output_is_deterministic() -> None:
    pdf = load("pdf_writer")
    first = pdf.render(_pages(), title="Supply Terms")
    assert first == pdf.render(_pages(), title="Supply Terms")
    assert first.startswith(b"%PDF-1.4") and first.endswith(b"%%EOF\n")
    assert b"/CreationDate" not in first and b"/ID" not in first


def test_long_text_wraps_and_what_does_not_fit_is_refused() -> None:
    pdf = load("pdf_writer")
    words = " ".join(["word"] * 200)
    assert len(pdf.wrap(words, "F1", 10.5, 468.0)) > 1
    assert all(
        pdf.text_width(line, "F1", 10.5) <= 468.0
        for line in pdf.wrap(words, "F1", 10.5, 468.0)
    )
    with pytest.raises(ValueError, match="fit on one page"):
        pdf.render([[pdf.Paragraph("\n".join(["line"] * 100))]])
    with pytest.raises(ValueError, match="wider than the page"):
        pdf.render([[pdf.Table(((words,),))]])
    with pytest.raises(ValueError, match="WinAnsi"):
        pdf.render([[pdf.Paragraph("계약")]])
    with pytest.raises(ValueError, match="at least one page"):
        pdf.render([])
