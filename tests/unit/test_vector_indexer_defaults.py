# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``VectorIndexer`` port's defaults a third-party store relies on."""

from __future__ import annotations

from typing import Any

import pytest

from unified_kg_rag.domain.models import Config
from unified_kg_rag.ports.indexer import VectorIndexer

pytestmark = pytest.mark.unit


class _Bare(VectorIndexer):
    """A store implementing none of the optional reads."""

    def __init__(self) -> None:
        super().__init__(Config())


_Bare.__abstractmethods__ = frozenset()


class _Scanning(_Bare):
    """A store implementing ``scan_text_units`` alone."""

    def __init__(self, units: list[dict[str, Any]]) -> None:
        super().__init__()
        self.units = units
        self.scans: list[tuple[str | None, list[str]]] = []

    def scan_text_units(
        self,
        suffix: str | None,
        *,
        fields: list[str],
        page_size: int | None = None,
    ) -> list[dict[str, Any]] | None:
        self.scans.append((suffix, fields))
        return self.units


def test_text_unit_pages_serve_a_store_that_only_scans() -> None:
    # Without it the summary tree stage found no pages and built no tree for
    # a store that implements only scan_text_units.
    units = [{"id": f"t{i}", "text": "벤더"} for i in range(5)]
    store = _Scanning(units)

    pages = store.scan_text_unit_pages("x", fields=["text"], page_size=2)

    assert pages is not None
    assert [[u["id"] for u in page] for page in pages] == [
        ["t0", "t1"],
        ["t2", "t3"],
        ["t4"],
    ]
    assert store.scans == [("x", ["text"])]
    whole = store.scan_text_unit_pages("x", fields=["text"])
    assert whole is not None and list(whole) == [units]
    empty = _Scanning([]).scan_text_unit_pages("x", fields=["text"])
    assert empty is not None and list(empty) == []


def test_a_store_that_cannot_scan_cannot_page() -> None:
    store = _Bare()
    assert store.scan_text_unit_pages("x", fields=["text"]) is None
    assert store.extraction_record_keys(["fp:"], suffix="x") is None
