# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A synthetic source corpus covering the parser's byte-level edge cases.

Line endings (CRLF, bare CR), byte-order marks, UTF-16 and cp949 text, CSV
quoting (embedded delimiters, quotes and newlines, short and long rows) and a
multi-page PDF with an info dictionary. The files are written byte for byte,
so a document's parsed text, and with it its content hash and
``document_id``, depends on the loaders alone.
"""

from __future__ import annotations

from pathlib import Path

KOREAN = "한국어 문서입니다. 공급자는 구매자에게 물품을 납품한다.\n둘째 줄입니다.\n"

# Name -> raw bytes. Text that needs a non-UTF-8 decoder exercises the
# detect-and-retry path of FileParser.
TEXT_FILES: dict[str, bytes] = {
    "crlf.txt": b"Vendor ships parts.\r\nBuyer pays.\r\n\r\nSecond paragraph.\r\n",
    "bare_cr.txt": b"Old line one.\rOld line two.\r",
    "mixed_newlines.txt": b"a line\r\nanother\nlast\r",
    "bom.txt": "\ufeffVendor with a BOM.\nSecond line.\n".encode(),
    "utf16.txt": ("Vendor in UTF-16.\r\n" + KOREAN).encode("utf-16"),
    "cp949.txt": KOREAN.encode("cp949"),
    "latin1.txt": "Café crème brûlée à la carte, "
    "naïve façade, déjà vu, über Straße.\n".encode("latin-1"),
    "trailing_ws.txt": b"  leading and trailing spaces  \n\n\n",
}

CSV_FILES: dict[str, bytes] = {
    "quotes.csv": (
        b"name,description,amount\r\n"
        b'Vendor,"Ships parts, tools and ""spares""",100\r\n'
        b'Buyer,"Pays on\r\ndelivery\nnext line",  250  \r\n'
        b"Short,row\r\n"
        b"Long,row,1,extra,fields\r\n"
        b'Empty,"",\r\n'
    ),
    "bom.csv": "\ufeffkey,value\nalpha,1\nbeta,2\n".encode(),
    "cp949.csv": "이름,설명\n공급자,물품 납품\n구매자,대금 지급\n".encode("cp949"),
}

PDF_PAGES = (
    ["Vendor agrees to ship 100 units.", "Delivery within 30 days."],
    ["Buyer pays on receipt.", "Late payment accrues interest."],
    ["Either party may terminate.", "Notice period: 60 days."],
)

MARKDOWN = (
    "# Supply terms\r\n\r\n"
    "The **Vendor** ships parts to the *Buyer*.\r\n\r\n"
    "- First item\r\n- Second item\r\n\r\n"
    "| Term | Value |\r\n|------|-------|\r\n| Days | 30 |\r\n"
)

HTML = (
    "<html><head><title>Terms</title></head><body>"
    "<h1>Supply terms</h1><p>The Vendor ships parts.</p>"
    "<ul><li>First</li><li>Second</li></ul>"
    "<table><tr><td>Days</td><td>30</td></tr></table>"
    "</body></html>"
)


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: tuple[list[str], ...] = PDF_PAGES) -> bytes:
    """A minimal PDF, one Helvetica text line per entry, with an info dict."""
    objects: list[bytes] = []
    page_ids = [3 + 2 * i for i in range(len(pages))]
    font_id = 3 + 2 * len(pages)
    info_id = font_id + 1
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    for pid, lines in zip(page_ids, pages, strict=True):
        body = "BT /F1 12 Tf 14 TL 72 720 Td " + " ".join(
            f"({_escape_pdf_text(line)}) Tj T*" for line in lines
        )
        stream = (body + " ET").encode()
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {pid + 1} 0 R >>".encode()
        )
        objects.append(
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream"
        )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(
        b"<< /Title (Synthetic supply terms) /Producer (fixture) "
        b"/Creator (loader_corpus) /CreationDate (D:20240102030405+09'00') >>"
    )
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += (
        b"trailer\n<< /Size %d /Root 1 0 R /Info %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
        % (len(objects) + 1, info_id, xref)
    )
    return bytes(out)


def write_loader_corpus(root: Path, *, markup: bool = False) -> list[Path]:
    """Write the corpus under ``root``; ``markup`` adds the .md/.html files
    (they need the optional ``unstructured`` extra to parse)."""
    files: dict[str, bytes] = {**TEXT_FILES, **CSV_FILES, "multi.pdf": build_pdf()}
    if markup:
        files["terms.md"] = MARKDOWN.encode()
        files["terms.html"] = HTML.encode()
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, data in sorted(files.items()):
        path = root / name
        path.write_bytes(data)
        paths.append(path)
    return paths
