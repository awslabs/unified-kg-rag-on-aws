# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for DocStatus models and the DocStatusPort diff contract (M1/M2)."""

from __future__ import annotations

import pytest

from aws_graphrag.domain.models import DocStatus, DocStatusRecord, DocumentDelta
from aws_graphrag.ports import DocStatusPort
from tests.fixtures.fakes.doc_status import FakeDocStatusStore

pytestmark = pytest.mark.unit


def test_record_defaults_to_pending() -> None:
    record = DocStatusRecord(doc_id="d1", content_hash="h1")
    assert record.status is DocStatus.PENDING
    assert record.entity_ids == []


def test_empty_delta_is_empty() -> None:
    assert DocumentDelta().is_empty is True


def test_delta_to_process_is_new_plus_changed() -> None:
    delta = DocumentDelta(new=["a"], changed=["b"], unchanged=["c"], deleted=["d"])
    assert delta.to_process == ["a", "b"]
    assert delta.is_empty is False


def test_fake_store_conforms_to_port() -> None:
    assert isinstance(FakeDocStatusStore(), DocStatusPort)


def test_fake_store_roundtrip(fake_doc_status: FakeDocStatusStore) -> None:
    record = DocStatusRecord(doc_id="d1", content_hash="h1", entity_ids=["e1"])
    fake_doc_status.put(record)
    assert fake_doc_status.get("d1") == record
    assert fake_doc_status.list_all() == [record]
    fake_doc_status.delete("d1")
    assert fake_doc_status.get("d1") is None


def test_diff_classifies_new_changed_unchanged_deleted(
    fake_doc_status: FakeDocStatusStore,
) -> None:
    fake_doc_status.put(DocStatusRecord(doc_id="keep", content_hash="h1"))
    fake_doc_status.put(DocStatusRecord(doc_id="edit", content_hash="old"))
    fake_doc_status.put(DocStatusRecord(doc_id="gone", content_hash="h3"))

    delta = fake_doc_status.diff({"keep": "h1", "edit": "new", "fresh": "h4"})

    assert delta.new == ["fresh"]
    assert delta.changed == ["edit"]
    assert delta.unchanged == ["keep"]
    assert delta.deleted == ["gone"]


def test_diff_empty_registry_marks_everything_new(
    fake_doc_status: FakeDocStatusStore,
) -> None:
    delta = fake_doc_status.diff({"a": "h", "b": "h"})
    assert set(delta.new) == {"a", "b"}
    assert delta.changed == [] and delta.deleted == []
