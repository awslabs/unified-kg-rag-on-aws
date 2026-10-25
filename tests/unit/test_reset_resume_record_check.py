# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The stage a reset resumed past the extraction-record stages starts from.

``DataIngestionPipeline._rewind_on_missing_records`` checks that the
extraction records the restored text units name are still stored, and moves
the resume back to the first stage whose records another pipeline id
replaced. The integration tests (``test_reset_rerun_records.py``) cover the
graph records on the embedded store; these cover the other branches against
a vector indexer built on the port, its ``stored_extraction_record_keys``
the port's default.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from unified_kg_rag.application.ingestion import pipeline as pipeline_module
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.ingestion.contributions import RECORD_KEYS_ATTRIBUTE
from unified_kg_rag.domain.models import Config, Constants, TextUnit
from unified_kg_rag.ports import VectorIndexer
from unified_kg_rag.shared.exceptions import NamespaceLeaseError

pytestmark = pytest.mark.unit

_ORDER = list(DataIngestionPipeline._ORDER)
_INDEXING = "indexing"
_DONE = _ORDER[: _ORDER.index(_INDEXING)]


class _Indexer(VectorIndexer):
    """A vector indexer holding the extraction records ``stored`` names
    (keys by suffix), or failing every read with ``stored``."""

    def __init__(self, stored: dict[str, set[str]] | Exception) -> None:
        super().__init__(Config())
        self.stored = stored
        self.closed = False

    def read_extraction_records(
        self, keys: list[str], suffix: str | None = None
    ) -> dict[str, dict[str, Any]]:
        if isinstance(self.stored, Exception):
            raise self.stored
        held = self.stored.get(suffix or Constants.DEFAULT_SUFFIX.value, set())
        return {key: {} for key in keys if key in held}

    def close(self) -> None:
        self.closed = True


# The writes this check never makes.
_Indexer.__abstractmethods__ = frozenset()


class _Keepless(_Indexer):
    """A backend that keeps no extraction records (the port's default)."""

    read_extraction_records = VectorIndexer.read_extraction_records  # type: ignore[assignment]


def _unit(unit_id: str, suffix: str) -> TextUnit:
    return TextUnit(
        id=unit_id,
        text="t",
        attributes={
            Constants.INDEX.value: suffix,
            RECORD_KEYS_ATTRIBUTE: {"graph": f"g-{unit_id}", "claims": f"c-{unit_id}"},
        },
    )


def _rewind(
    monkeypatch: pytest.MonkeyPatch, indexer: _Indexer, units: list[TextUnit]
) -> tuple[str, list[str]] | None:
    """Where a reset resumed at indexing, every stage before it restored,
    starts against ``indexer`` (None: at indexing)."""
    # The units as restored: they already name their records.
    monkeypatch.setattr(pipeline_module, "_with_record_keys", lambda _c, _x, u: u)
    pipeline = object.__new__(DataIngestionPipeline)
    pipeline.stages = [SimpleNamespace(name=name) for name in _ORDER]  # type: ignore[misc]
    pipeline.config = SimpleNamespace(indexing=SimpleNamespace(reset=True))  # type: ignore[assignment]
    pipeline.providers = SimpleNamespace(build_vector_indexer=lambda: indexer)  # type: ignore[assignment]
    context = SimpleNamespace(translated_units=None, text_units=units)
    rewound = pipeline._rewind_on_missing_records(
        context, _INDEXING, list(_DONE)  # type: ignore[arg-type]
    )
    assert indexer.closed
    return rewound


def _to(stage: str) -> tuple[str, list[str]]:
    return stage, _DONE[: _DONE.index(stage)]


_UNITS = [_unit("u0", "x"), _unit("u1", "y")]
_ALL = {"x": {"g-u0", "c-u0"}, "y": {"g-u1", "c-u1"}}


def test_records_still_stored_keep_the_start(monkeypatch) -> None:
    # Each suffix's records are read from its own store: read from another,
    # every record looked gone and the resume ran every stage again.
    assert _rewind(monkeypatch, _Indexer(_ALL), _UNITS) is None


def test_gone_claim_records_rewind_to_claim_extraction(monkeypatch) -> None:
    stored = {**_ALL, "y": {"g-u1"}}
    assert _rewind(monkeypatch, _Indexer(stored), _UNITS) == _to("claim_extraction")


def test_gone_graph_records_rewind_to_graph_extraction(monkeypatch) -> None:
    # Graph records are stored at graph_resolution but extracted before it.
    stored = {**_ALL, "x": {"c-u0"}}
    assert _rewind(monkeypatch, _Indexer(stored), _UNITS) == _to("graph_extraction")


def test_a_backend_keeping_no_records_keeps_the_start(monkeypatch) -> None:
    # It never stored any, so none is gone (asked, every record looks gone).
    assert _rewind(monkeypatch, _Keepless({}), _UNITS) is None


def test_a_store_error_leaves_the_check_to_the_indexing_stage(monkeypatch) -> None:
    indexer = _Indexer(RuntimeError("store down"))
    assert _rewind(monkeypatch, indexer, _UNITS) is None


def test_a_lost_namespace_lease_stops_the_resume(monkeypatch) -> None:
    # Another run holds the namespace: resuming against it must not go on.
    indexer = _Indexer(NamespaceLeaseError("lease lost"))
    with pytest.raises(NamespaceLeaseError):
        _rewind(monkeypatch, indexer, _UNITS)
    assert indexer.closed


def test_the_port_default_names_the_keys_the_store_reads() -> None:
    # An adapter overriding only read_extraction_records.
    indexer = _Indexer({"x": {"k1"}})
    assert indexer.stored_extraction_record_keys(["k1", "k2"], suffix="x") == {"k1"}
    assert indexer.stored_extraction_record_keys(["k1"], suffix="y") == set()
