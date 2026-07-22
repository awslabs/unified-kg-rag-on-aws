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

from unified_kg_rag.adapters.ingestion.parser import (
    _UNSTRUCTURED_AVAILABLE,
    ParserFactory,
)
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import DataProcessingError

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


# --- register_loader: custom-format extension seam --------------------------

from langchain_community.document_loaders.base import BaseLoader  # noqa: E402


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
    with pytest.raises(ValueError, match="leading dot"):
        ParserFactory.register_loader("xyz", _DummyLoader)


def test_register_loader_rejects_non_loader(_restore_loader_configs) -> None:
    with pytest.raises(TypeError, match="BaseLoader subclass"):
        ParserFactory.register_loader(".xyz", dict)  # type: ignore[arg-type]
