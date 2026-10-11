# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fakes and the local adapters must keep the exact call shape of their ports.

They are duck-typed (they do not subclass the ports), so nothing else stops a
port parameter rename from leaving one behind: tests would keep passing
against a call shape the real adapters no longer accept.
"""

from __future__ import annotations

import inspect

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.adapters.local.sqlite_doc_status import SQLiteDocStatusStore
from unified_kg_rag.ports.doc_status import DocStatusPort
from unified_kg_rag.ports.indexer import GraphIndexer, VectorIndexer


def _port_methods(port: type) -> list[str]:
    """Public instance methods of ``port`` (class/static helpers excluded)."""
    return sorted(
        name
        for name in dir(port)
        if not name.startswith("_")
        and inspect.isfunction(inspect.getattr_static(port, name))
    )


def _shape(func: object) -> list[tuple[str, inspect._ParameterKind, bool]]:
    return [
        (p.name, p.kind, p.default is not inspect.Parameter.empty)
        for p in inspect.signature(func).parameters.values()  # type: ignore[arg-type]
    ]


PAIRS = [
    (FakeGraphStore, GraphIndexer),
    (FakeVectorStore, VectorIndexer),
    (FakeDocStatusStore, DocStatusPort),
    (SQLiteDocStatusStore, DocStatusPort),
]


@pytest.mark.parametrize(
    ("fake", "port", "method"),
    [
        pytest.param(fake, port, method, id=f"{fake.__name__}.{method}")
        for fake, port in PAIRS
        for method in _port_methods(port)
    ],
)
def test_fake_matches_port_signature(fake: type, port: type, method: str) -> None:
    assert hasattr(fake, method), f"{fake.__name__} lacks port method {method}"
    assert _shape(getattr(fake, method)) == _shape(getattr(port, method))


@pytest.mark.parametrize(("fake", "port"), PAIRS)
def test_port_surface_is_not_empty(fake: type, port: type) -> None:
    # Guards the discovery above: an empty list would make the parametrized
    # test vacuously pass.
    assert len(_port_methods(port)) >= 5
