# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A UTF-8 file reads as UTF-8 under any locale (``file_loaders``).

The text loaders read with ``open()``'s default, the locale's preferred
encoding, when given none: under a Latin-1 locale a UTF-8 Korean file
decoded without an error, as mojibake, so the parser's detection never ran
and the source digest's claim (a UTF-8 file is read as UTF-8 either way)
did not hold.
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - runs this interpreter on a test script
import sys
from pathlib import Path
from typing import Any

import pytest

from unified_kg_rag.adapters.ingestion import file_loaders
from unified_kg_rag.adapters.ingestion.file_loaders import (
    DEFAULT_ENCODING,
    CsvFileLoader,
    HtmlFileLoader,
    TextFileLoader,
)
from unified_kg_rag.adapters.ingestion.parser import ParserFactory
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.unit

_KOREAN = "벤더는 구매자에게 부품을 공급한다. Café ünd naïve."


@pytest.mark.parametrize("loader", [TextFileLoader, CsvFileLoader, HtmlFileLoader])
def test_a_loader_without_an_encoding_reads_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, loader: type[Any]
) -> None:
    path = tmp_path / "doc.txt"
    path.write_text(f"name\n{_KOREAN}\n", encoding="utf-8")
    encodings: list[str | None] = []
    real_open = open

    def recording(file: Any, *args: Any, encoding: str | None = None, **kw: Any) -> Any:
        encodings.append(encoding)
        return real_open(file, *args, encoding=encoding, **kw)

    monkeypatch.setattr(file_loaders, "open", recording, raising=False)
    documents = loader(path).load()
    assert encodings == [DEFAULT_ENCODING] == ["utf-8-sig"]
    assert _KOREAN in "\n".join(d.page_content for d in documents)
    # A given encoding is read in.
    encodings.clear()
    loader(path, encoding="latin-1").load()
    assert encodings == ["latin-1"]


_SCRIPT = """
import sys
from pathlib import Path
from unified_kg_rag.adapters.ingestion.parser import ParserFactory
from unified_kg_rag.domain.models import Config
import locale
print(locale.getpreferredencoding(False), file=sys.stderr)
path = Path(sys.argv[1])
document = ParserFactory.create_parser(path, Config()).parse_file(path)
sys.stdout.buffer.write(document.page_content.encode("utf-8"))
"""


def _latin1_locale() -> str | None:
    """A Latin-1 locale this host has, or ``None``."""
    try:
        listed = subprocess.run(  # nosec B603 B607 - fixed argv
            ["locale", "-a"], capture_output=True, text=True, check=True
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return None
    for name in ("en_US.ISO-8859-1", "en_US.iso88591", "en_GB.iso885915"):
        if name in listed:
            return name
    return next((n for n in listed if "8859" in n.lower()), None)


def test_a_utf8_file_parses_alike_under_a_latin1_locale(tmp_path: Path) -> None:
    locale_name = _latin1_locale()
    if locale_name is None:
        pytest.skip("no Latin-1 locale on this host (locale -a)")
    path = tmp_path / "contract.txt"
    path.write_text(_KOREAN, encoding="utf-8")
    env = {
        **os.environ,
        "LC_ALL": locale_name,
        "LANG": locale_name,
        "PYTHONUTF8": "0",
        "PYTHONCOERCECLOCALE": "0",
    }
    result = subprocess.run(  # nosec B603 - this interpreter, fixed script
        [sys.executable, "-c", _SCRIPT, str(path)],
        capture_output=True,
        env=env,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert "8859" in result.stderr.decode("utf-8", "replace")
    assert result.stdout.decode("utf-8") == _KOREAN


_BOM = b"\xef\xbb\xbf"


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("vendors.csv", "name,note\n조르빅스,부품\nゾルビクス,部品\n"),
        (
            "contract.txt",
            "조르빅스와 Vendor는 부품을 거래한다.\n\nゾルビクスは南部にある。\n",
        ),
        ("page.html", "<html><body><p>조르빅스</p><p>ゾルビクス</p></body></html>"),
    ],
)
def test_a_byte_order_mark_is_not_text(tmp_path: Path, name: str, text: str) -> None:
    # rv27: Excel's "CSV UTF-8" (common for Korean and Japanese exports)
    # starts with a byte-order mark; read as plain UTF-8 it stayed in the
    # first header of every row after the first ("\ufeffname: ..."), as the
    # converter strips only a leading one. The file parses as it does
    # without the mark, and a file without one parses as plain UTF-8 reads it.
    plain, marked = tmp_path / "plain", tmp_path / "marked"
    plain.mkdir()
    marked.mkdir()
    (plain / name).write_bytes(text.encode("utf-8"))
    (marked / name).write_bytes(_BOM + text.encode("utf-8"))
    config = Config()

    def parsed(path: Path) -> str:
        return ParserFactory.create_parser(path, config).parse_file(path).page_content

    without = parsed(plain / name)
    assert "\ufeff" not in parsed(marked / name)
    assert parsed(marked / name) == without
    assert ("조르빅스" in without) and ("ゾルビクス" in without)
    if name.endswith(".csv"):
        assert without.count("name: ") == 2, without
