# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The filesystem CacheManager conforms to CachePort (AWS-free)."""

from __future__ import annotations

import pytest

from unified_kg_rag.ports import CachePort
from unified_kg_rag.shared.cache_manager import CacheManager

pytestmark = pytest.mark.unit

_PORT_METHODS = (
    "cache_exists",
    "save_stage_result",
    "load_stage_result",
    "get_pipeline_cache_dir",
    "load_cache_index",
    "get_cache_stats",
)


def test_cache_manager_has_port_methods() -> None:
    # Structural conformance: every CachePort method exists on CacheManager.
    for method in _PORT_METHODS:
        assert callable(
            getattr(CacheManager, method, None)
        ), f"CacheManager missing {method}"


def test_cache_manager_is_recognized_as_cache_port() -> None:
    # runtime_checkable structural typing accepts the concrete manager...
    assert issubclass(CacheManager, CachePort)


def test_runtime_checkable_protocol_rejects_non_conforming() -> None:
    class _NotACache:
        pass

    assert not isinstance(_NotACache(), CachePort)
