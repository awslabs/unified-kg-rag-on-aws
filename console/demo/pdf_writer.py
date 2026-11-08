# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A minimal PDF writer for the console demo's corpus (standard library only).

It lays out pages of headings, paragraphs and ruled tables in the base-14
Helvetica and Helvetica-Bold fonts (WinAnsi text, no embedded font), so the
demo ships PDFs without a binary in the repository or a new dependency.
The output is deterministic: no creation date, no document id, the same
bytes for the same pages. Text a parser extracts comes out one line per
laid-out line (``pypdf``: a table row is one line of its cells).

    pdf = render([[Heading("Title"), Paragraph("One sentence.")]], title="Demo")
"""

from __future__ import annotations

from dataclasses import dataclass

PAGE_WIDTH = 612.0  # US Letter, in points
PAGE_HEIGHT = 792.0
MARGIN = 72.0

# Advance widths (1/1000 em) of the printable ASCII characters, from the
# Adobe font metrics of the base-14 fonts; other characters use DEFAULT_WIDTH.
_HELVETICA = (
    "278 278 355 556 556 889 667 191 333 333 389 584 278 333 278 278 "
    "556 556 556 556 556 556 556 556 556 556 278 278 584 584 584 556 "
    "1015 667 667 722 722 667 611 778 722 278 500 667 556 833 722 778 "
    "667 778 722 667 611 722 667 944 667 667 611 278 278 278 469 556 "
    "333 556 556 500 556 556 278 556 556 222 222 500 222 833 556 556 "
    "556 556 333 500 278 556 500 722 500 500 500 334 260 334 584"
)
_HELVETICA_BOLD = (
    "278 333 474 556 556 889 722 238 333 333 389 584 278 333 278 278 "
    "556 556 556 556 556 556 556 556 556 556 333 333 584 584 584 611 "
    "975 722 722 722 722 667 611 778 722 278 556 722 611 833 722 778 "
    "667 778 722 667 611 722 667 944 667 667 611 333 278 333 584 556 "
    "333 556 611 556 611 556 333 611 611 278 278 556 278 889 611 611 "
    "611 611 389 556 333 611 556 778 556 556 500 389 280 389 584"
)
DEFAULT_WIDTH = 556
_WIDTHS = {
    "F1": dict(
        zip(map(chr, range(32, 127)), map(int, _HELVETICA.split()), strict=True)
    ),
    "F2": dict(
        zip(map(chr, range(32, 127)), map(int, _HELVETICA_BOLD.split()), strict=True)
    ),
}
_FONT_NAMES = {"F1": "Helvetica", "F2": "Helvetica-Bold"}


@dataclass(frozen=True)
class Heading:
    """A bold line, ``size`` points high."""

    text: str
    size: float = 14.0


@dataclass(frozen=True)
class Paragraph:
    """Text wrapped to the page width; each ``\\n`` starts a new line."""

    text: str
    size: float = 10.5
    bold: bool = False


@dataclass(frozen=True)
class Table:
    """A ruled table: the first row is the header (bold, shaded)."""

    rows: tuple[tuple[str, ...], ...]
    size: float = 10.0


Block = Heading | Paragraph | Table


def text_width(text: str, font: str, size: float) -> float:
    """``text``'s width in points in ``font`` (``F1`` or ``F2``)."""
    widths = _WIDTHS[font]
    return sum(widths.get(c, DEFAULT_WIDTH) for c in text) * size / 1000.0


def wrap(text: str, font: str, size: float, width: float) -> list[str]:
    """``text``'s lines: each ``\\n`` breaks one, and words wrap at ``width``."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}" if line else word
            if line and text_width(candidate, font, size) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def _literal(text: str) -> str:
    """``text`` as a PDF literal string in WinAnsi (cp1252) encoding."""
    try:
        raw = text.encode("cp1252")
    except UnicodeEncodeError as e:
        raise ValueError(
            f"{text!r} is not WinAnsi text (the base-14 fonts are Latin-only)"
        ) from e
    out = []
    for byte in raw:
        char = chr(byte)
        if char in "()\\":
            out.append("\\" + char)
        elif 32 <= byte < 127:
            out.append(char)
        else:
            out.append(f"\\{byte:03o}")
    return "(" + "".join(out) + ")"


def _number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


class _Page:
    """One page's content stream, laid out top-down."""

    def __init__(self) -> None:
        self.ops: list[str] = []
        self.y = PAGE_HEIGHT - MARGIN

    def advance(self, height: float) -> None:
        self.y -= height
        if self.y < MARGIN:
            raise ValueError("the blocks do not fit on one page; split them")

    def text(self, x: float, y: float, text: str, font: str, size: float) -> None:
        self.ops.append(
            f"BT /{font} {_number(size)} Tf {_number(x)} {_number(y)} Td "
            f"{_literal(text)} Tj ET"
        )

    def line(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.ops.append(
            f"{_number(x1)} {_number(y1)} m {_number(x2)} {_number(y2)} l S"
        )

    def heading(self, block: Heading) -> None:
        self.advance(block.size * 1.2)
        self.text(MARGIN, self.y, block.text, "F2", block.size)
        self.advance(block.size * 0.5)

    def paragraph(self, block: Paragraph) -> None:
        font = "F2" if block.bold else "F1"
        for line in wrap(block.text, font, block.size, PAGE_WIDTH - 2 * MARGIN):
            self.advance(block.size * 1.4)
            self.text(MARGIN, self.y, line, font, block.size)
        self.advance(block.size * 0.6)

    def table(self, block: Table) -> None:
        size, pad = block.size, 6.0
        columns = max(len(row) for row in block.rows)
        widths = [
            max(
                text_width(
                    row[i] if i < len(row) else "", "F2" if r == 0 else "F1", size
                )
                for r, row in enumerate(block.rows)
            )
            + 2 * pad
            for i in range(columns)
        ]
        total = sum(widths)
        if total > PAGE_WIDTH - 2 * MARGIN:
            raise ValueError("the table is wider than the page")
        height = size * 1.8
        self.advance(size * 0.4)
        top = self.y
        self.ops.append("0.5 w")
        for r, row in enumerate(block.rows):
            bottom = self.y - height
            if bottom < MARGIN:
                raise ValueError("the blocks do not fit on one page; split them")
            if r == 0:
                self.ops.append(
                    f"0.9 g {_number(MARGIN)} {_number(bottom)} {_number(total)} "
                    f"{_number(height)} re f 0 g"
                )
            x = MARGIN
            for i, width in enumerate(widths):
                cell = row[i] if i < len(row) else ""
                if cell:
                    font = "F2" if r == 0 else "F1"
                    self.text(x + pad, bottom + size * 0.6, cell, font, size)
                x += width
            self.line(MARGIN, bottom, MARGIN + total, bottom)
            self.y = bottom
        self.line(MARGIN, top, MARGIN + total, top)
        x = MARGIN
        for width in [0.0, *widths]:
            x += width
            self.line(x, top, x, self.y)
        self.advance(size * 1.2)


def render(pages: list[list[Block]], *, title: str | None = None) -> bytes:
    """A PDF of ``pages`` (each a list of blocks laid out top-down)."""
    if not pages:
        raise ValueError("a PDF needs at least one page")
    streams = []
    for blocks in pages:
        page = _Page()
        for block in blocks:
            if isinstance(block, Heading):
                page.heading(block)
            elif isinstance(block, Paragraph):
                page.paragraph(block)
            else:
                page.table(block)
        streams.append("\n".join(page.ops).encode("latin-1"))

    # Objects: 1 catalog, 2 page tree, 3-4 fonts, 5 info, then a page and
    # its content stream per page.
    objects: list[bytes] = []
    kids = " ".join(f"{6 + 2 * i} 0 R" for i in range(len(pages)))
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("latin-1")
    )
    for key in ("F1", "F2"):
        objects.append(
            f"<< /Type /Font /Subtype /Type1 /BaseFont /{_FONT_NAMES[key]} "
            "/Encoding /WinAnsiEncoding >>".encode("latin-1")
        )
    info = "/Producer (console demo pdf_writer)"
    if title:
        info = f"/Title {_literal(title)} " + info
    objects.append(f"<< {info} >>".encode("latin-1"))
    for i, stream in enumerate(streams):
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R "
                f"/MediaBox [0 0 {_number(PAGE_WIDTH)} {_number(PAGE_HEIGHT)}] "
                "/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> "
                f"/Contents {7 + 2 * i} 0 R >>"
            ).encode("latin-1")
        )
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1")
            + stream
            + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode("latin-1") + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info 5 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)
