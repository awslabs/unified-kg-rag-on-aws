# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run whose stores reject item writes never makes its documents re-extract.

Random corpora: one is indexed, the next run over another has some of its
entity, relationship or claim writes rejected by one store, and the run
after that sees the same corpus with the stores healthy again. The rejected
run records no document FAILED (the items are noted for the next commit,
see ``application.storage.pending_writes``); the follow-up extracts only
what the rejected run left unrecorded (over the tolerated failure rate) and
leaves exactly what a fresh full build stores.
"""

from __future__ import annotations

import hashlib

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.fixtures.doc_status_records import all_records
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.incremental_runs import Corpus, IncrementalRun, expected_state
from unified_kg_rag.domain.ingestion.names import compact_name_key
from unified_kg_rag.domain.models import Config, DocStatus

pytestmark = pytest.mark.property

_PATHS = ["/corpus/a.txt", "/corpus/b.txt", "/corpus/c.txt", "/corpus/d.txt"]
# With spellings of one entity (legal forms the full build folds): which
# one an entity is stored under follows the corpus, so a commit re-keys it
# and its edges.
_NAMES = [
    "Vendor",
    "Depot",
    "Carrier",
    "Buyer",
    "Acme Inc",
    "Acme Corp",
    "Acme",
    "주식회사 한빛",
    "한빛",
]

_edge = st.tuples(
    st.sampled_from(_NAMES), st.sampled_from(_NAMES), st.integers(1, 3)
).filter(lambda e: compact_name_key(e[0]) != compact_name_key(e[1]))
_unit = st.lists(_edge, min_size=1, max_size=3, unique_by=lambda e: (e[0], e[1]))
_content = st.lists(_unit, min_size=1, max_size=2)
_corpus = st.fixed_dictionaries({path: st.none() | _content for path in _PATHS}).map(
    lambda state: {path: content for path, content in state.items() if content}
)

_WRITES = [
    "graph.upsert_entities",
    "graph.upsert_relationships",
    "vector.upsert_entities",
    "vector.upsert_relationships",
    "vector.update_entity_lineage",
    "vector.update_relationship_lineage",
    "vector.upsert_claims",
]


def _rejects(where: str, salt: int):  # noqa: ANN202
    def fail(at: str, item_id: str) -> bool:
        digest = hashlib.sha256(f"{salt}:{item_id}".encode()).digest()
        return at == where and digest[0] % 3 == 0

    return fail


# Over 120 s (the default timeout) on a loaded runner: the timeout leaves
# margin for that, not for a hang.
@pytest.mark.slow
@pytest.mark.timeout(600)
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    before=_corpus,
    after=_corpus,
    where=st.sampled_from(_WRITES),
    salt=st.integers(0, 1000),
)
def test_rejected_writes_are_written_by_the_next_run(
    before: Corpus, after: Corpus, where: str, salt: int
) -> None:
    config = Config()
    config.indexing.max_failure_rate = 0.9
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run(before)

    harness.run(after, fail=_rejects(where, salt))

    statuses = {r.status for r in all_records(harness.store)}
    assert DocStatus.FAILED not in statuses
    unrecorded = sorted(
        r.file_path or r.doc_id
        for r in all_records(harness.store)
        if r.status is not DocStatus.PROCESSED
    )
    harness.run(after)
    assert harness.extracted == [p for p in unrecorded if p in after]
    assert harness.state() == expected_state(after, config)
    assert harness.registry_problems(after) == []
    harness.run(after)
    assert harness.extracted == []
