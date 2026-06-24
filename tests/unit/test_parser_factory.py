# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ParserFactory format support + optional-dependency gating.

Regression: .md/.html were advertised as supported but their loaders need the
optional `unstructured` package, which is absent from the data-plane image —
so a .md source failed mid-ingestion with "No module named 'unstructured'"
(found during incremental-indexing E2E). The factory now registers those
formats only when `unstructured` is importable.
"""

from __future__ import annotations

import pytest

from aws_graphrag.adapters.ingestion.parser import (
    _UNSTRUCTURED_AVAILABLE,
    ParserFactory,
)
from aws_graphrag.domain.models import Config
from aws_graphrag.shared import DataProcessingError

pytestmark = pytest.mark.unit


def test_core_formats_always_supported() -> None:
    exts = set(ParserFactory.get_supported_extensions())
    # No-heavy-dep loaders must always be available.
    assert {".txt", ".csv", ".json", ".pdf"} <= exts


def test_markdown_gated_on_unstructured() -> None:
    exts = set(ParserFactory.get_supported_extensions())
    if _UNSTRUCTURED_AVAILABLE:
        assert {".md", ".markdown", ".html", ".htm"} <= exts
    else:
        # Must NOT advertise formats it cannot actually parse.
        assert not ({".md", ".markdown", ".html", ".htm"} & exts)


def test_unsupported_type_raises() -> None:
    with pytest.raises(DataProcessingError, match="Unsupported file type"):
        ParserFactory.create_parser("data.xyz", Config())


@pytest.mark.skipif(
    _UNSTRUCTURED_AVAILABLE, reason="unstructured installed; .md is supported"
)
def test_markdown_error_hints_at_optional_dep() -> None:
    with pytest.raises(DataProcessingError, match="unstructured"):
        ParserFactory.create_parser("notes.md", Config())


def test_core_parser_constructs() -> None:
    # A supported core format yields a parser without touching AWS.
    parser = ParserFactory.create_parser("doc.txt", Config())
    assert parser is not None
