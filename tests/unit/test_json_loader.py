# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON source parsing without the optional ``jq`` package.

Regression: ``.json`` was registered with LangChain's ``JSONLoader``, which
needs ``jq`` (not a dependency), so every JSON source failed to parse.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unified_kg_rag.adapters.ingestion.json_loader import JsonTextLoader
from unified_kg_rag.adapters.ingestion.parser import FileParser, ParserFactory
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import DataProcessingError

pytestmark = pytest.mark.unit

_RECORD = {
    "title": "Supply Agreement",
    "parties": {"vendor": "Vendor Co", "buyer": "Buyer Ltd"},
    "amount": 1000,
    "notes": ["Delivery within 30 days", "Payment on invoice"],
}


def _write(path: Path, data: object, encoding: str = "utf-8") -> Path:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding=encoding)
    return path


def test_object_parses_via_factory(tmp_path: Path) -> None:
    path = _write(tmp_path / "agreement.json", _RECORD)
    parser = ParserFactory.create_parser(path, Config())
    assert isinstance(parser, FileParser)
    assert parser.loader_class is JsonTextLoader

    doc = parser.parse_file(path)
    text = doc.content.text
    assert '"vendor": "Vendor Co"' in text  # keys kept for extraction context
    assert "Delivery within 30 days" in text
    assert doc.file_type == "json"
    assert doc.total_pages == 1
    assert doc.metadata["source"] == str(path)


def test_array_yields_one_page_per_element(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "records.json",
        [{"name": "Vendor Co"}, {"name": "Buyer Ltd"}, "A plain note"],
    )
    docs = JsonTextLoader(path).load()
    assert [d.metadata["seq_num"] for d in docs] == [1, 2, 3]
    assert docs[2].page_content == "A plain note"


def test_strings_mode_keeps_only_string_leaves(tmp_path: Path) -> None:
    path = _write(tmp_path / "agreement.json", _RECORD)
    (doc,) = JsonTextLoader(path, text_mode="strings").load()
    assert doc.page_content.splitlines() == [
        "Supply Agreement",
        "Vendor Co",
        "Buyer Ltd",
        "Delivery within 30 days",
        "Payment on invoice",
    ]


def test_non_ascii_preserved_and_bom_accepted(tmp_path: Path) -> None:
    path = _write(tmp_path / "ko.json", {"당사자": "갑"}, encoding="utf-8-sig")
    (doc,) = JsonTextLoader(path).load()
    assert '"당사자": "갑"' in doc.page_content


def test_non_utf8_file_retried_with_detected_encoding(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "cp949.json",
        {"설명": "공급자는 구매자에게 부품을 납품한다"},
        encoding="cp949",
    )
    doc = ParserFactory.create_parser(path, Config()).parse_file(path)
    assert "공급자" in doc.content.text


def test_invalid_json_fails_with_processing_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(DataProcessingError):
        ParserFactory.create_parser(path, Config()).parse_file(path)


def test_empty_payload_fails_as_empty(tmp_path: Path) -> None:
    path = _write(tmp_path / "empty.json", [])
    with pytest.raises(DataProcessingError):
        ParserFactory.create_parser(path, Config()).parse_file(path)


def test_rejects_unknown_text_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        JsonTextLoader(tmp_path / "x.json", text_mode="flat")  # type: ignore[arg-type]
