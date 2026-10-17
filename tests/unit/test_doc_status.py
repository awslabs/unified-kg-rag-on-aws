# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DocStatus models (the port contract: doc-status conformance)."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.models import DocStatus, DocStatusRecord, DocumentDelta

pytestmark = pytest.mark.unit


def test_record_defaults_to_pending() -> None:
    record = DocStatusRecord(doc_id="d1", content_hash="h1")
    assert record.status is DocStatus.PENDING
    assert record.text_unit_ids == []


def test_empty_delta_is_empty() -> None:
    assert DocumentDelta().is_empty is True


def test_delta_to_process_is_new_plus_changed() -> None:
    delta = DocumentDelta(new=["a"], changed=["b"], unchanged=["c"], deleted=["d"])
    assert delta.to_process == ["a", "b"]
    assert delta.is_empty is False
