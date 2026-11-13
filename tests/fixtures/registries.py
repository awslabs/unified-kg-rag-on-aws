# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolating a module-level ``Registry`` from a test's registrations."""

from __future__ import annotations

from typing import Any

import pytest

from unified_kg_rag.adapters.ingestion import parser as parser_module
from unified_kg_rag.shared.registry import Registry


def isolate(monkeypatch: pytest.MonkeyPatch, registry: Registry[Any]) -> None:
    """Give ``registry`` copies of its tables until the test ends: what the
    test registers or removes is undone."""
    monkeypatch.setattr(registry, "entries", dict(registry.entries))
    monkeypatch.setattr(registry, "requires", dict(registry.requires))


def isolate_loaders(monkeypatch: pytest.MonkeyPatch) -> Registry[Any]:
    """``isolate`` the file-loader registry (``ParserFactory``); return it."""
    isolate(monkeypatch, parser_module._LOADERS)
    return parser_module._LOADERS
