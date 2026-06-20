# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Property-based tests for hashing and DocStatusPort.diff invariants."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from aws_graphrag.domain.models import DocStatusRecord
from aws_graphrag.utils.common import compute_hash
from tests.fixtures.fakes.doc_status import FakeDocStatusStore

pytestmark = pytest.mark.property


@given(st.text())
def test_compute_hash_is_deterministic(data: str) -> None:
    assert compute_hash(data) == compute_hash(data)


@given(st.text(), st.integers(min_value=1, max_value=64))
def test_compute_hash_length_is_bounded(data: str, length: int) -> None:
    assert len(compute_hash(data, length=length)) == length


# doc_id -> content_hash maps; ids are short tokens, hashes arbitrary text.
_ids = st.text(alphabet="abcdefghij", min_size=1, max_size=4)
_corpus = st.dictionaries(_ids, st.text(min_size=1, max_size=8), max_size=10)


@given(stored=_corpus, incoming=_corpus)
def test_diff_partitions_are_disjoint_and_complete(
    stored: dict[str, str], incoming: dict[str, str]
) -> None:
    store = FakeDocStatusStore()
    for doc_id, content_hash in stored.items():
        store.put(DocStatusRecord(doc_id=doc_id, content_hash=content_hash))

    delta = store.diff(incoming)

    # new + changed + unchanged exactly partitions the incoming keys.
    classified = set(delta.new) | set(delta.changed) | set(delta.unchanged)
    assert classified == set(incoming)
    assert len(delta.new) + len(delta.changed) + len(delta.unchanged) == len(incoming)
    # deleted are exactly the stored ids absent from incoming.
    assert set(delta.deleted) == set(stored) - set(incoming)


@given(corpus=_corpus)
def test_diff_unchanged_when_corpus_identical(corpus: dict[str, str]) -> None:
    store = FakeDocStatusStore()
    for doc_id, content_hash in corpus.items():
        store.put(DocStatusRecord(doc_id=doc_id, content_hash=content_hash))

    delta = store.diff(corpus)

    assert delta.is_empty is True
    assert set(delta.unchanged) == set(corpus)
