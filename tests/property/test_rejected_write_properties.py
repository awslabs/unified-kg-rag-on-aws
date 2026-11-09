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
from hypothesis import HealthCheck, example, given, settings
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
# A noted claim joined alone is derived again with every statement of its
# unit: one naming "Acme Inc" beside a stored "Acme" (one name key, two
# entities) resolved to "Acme" without the stored spelling, a claim the
# full build does not have.
@example(
    before={},
    after={
        "/corpus/c.txt": [[("Depot", "Acme Inc", 2)]],
        "/corpus/d.txt": [
            [("Depot", "Acme Inc", 2)],
            [("Acme", "주식회사 한빛", 3), ("한빛", "Acme Inc", 3)],
        ],
    },
    where="vector.upsert_claims",
    salt=1,
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


# The run after the rejected one sees another corpus, or the same one with
# one or two documents removed: a removal derives the stored items while the
# rejected ones are still only noted.
_mid_and_after = _corpus.flatmap(
    lambda mid: st.tuples(
        st.just(mid),
        _corpus
        | st.sets(st.sampled_from(_PATHS), min_size=1, max_size=2).map(
            lambda gone: {p: c for p, c in mid.items() if p not in gone}
        ),
    )
)


@pytest.mark.slow
@pytest.mark.timeout(600)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    before=_corpus,
    corpora=_mid_and_after,
    where=st.sampled_from(_WRITES),
    salt=st.integers(0, 1000),
)
# The noted edges at "Acme" (Acme Inc merged into it) are written by a run
# whose removal re-keys the entity to "Acme Inc" while its own write of them
# is rejected again: noted at an id no entity has, every later run failed
# them ("endpoint entity not found").
@example(
    before={},
    corpora=(
        {
            "/corpus/a.txt": [[("Depot", "Acme Inc", 2), ("Acme Inc", "Depot", 1)]],
            "/corpus/d.txt": [[("Buyer", "Acme", 1)]],
        },
        {"/corpus/a.txt": [[("Depot", "Acme Inc", 2), ("Acme Inc", "Depot", 1)]]},
    ),
    where="graph.upsert_entities",
    salt=150,
)
# The removal of b.txt derives the claim about "Acme Corp" while "Acme"
# (a.txt) is only noted: resolved without it, the claim's object was "Acme
# Corp", an entity no store holds, and no later run touched the claim.
@example(
    before={},
    corpora=(
        {
            "/corpus/a.txt": [[("Depot", "Acme", 1)]],
            "/corpus/b.txt": [[("Depot", "Buyer", 1), ("Buyer", "Acme Inc", 3)]],
            "/corpus/c.txt": [[("Buyer", "Acme Corp", 3)]],
        },
        {
            "/corpus/a.txt": [[("Depot", "Acme", 1)]],
            "/corpus/c.txt": [[("Buyer", "Acme Corp", 3)]],
        },
    ),
    where="graph.upsert_entities",
    salt=10,
)
def test_writes_rejected_twice_are_written_by_the_next_run(
    before: Corpus, corpora: tuple[Corpus, Corpus], where: str, salt: int
) -> None:
    mid, after = corpora
    config = Config()
    config.indexing.max_failure_rate = 0.9
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run(before)
    harness.run(mid, fail=_rejects(where, salt))
    harness.run(after, fail=_rejects(where, salt + 1))

    harness.run(after)

    assert harness.state() == expected_state(after, config)
    assert harness.registry_problems(after) == []


# One entity's spellings per language (legal forms the full build folds):
# a corpus of a few names and one language's spellings, so a run folds,
# splits and re-keys the noted entities often.
_SPELLINGS = {
    "en": ["Acme Inc", "Acme Corp", "Acme", "ACME"],
    "ko": ["주식회사 한빛", "한빛", "(주)한빛", "한빛 주식회사"],
    "ja": ["株式会社ひかり", "ひかり", "ひかり株式会社", "(株)ひかり"],
}


def _corpus_of(names: list[str]) -> st.SearchStrategy[Corpus]:
    edge = st.tuples(
        st.sampled_from(names), st.sampled_from(names), st.integers(1, 3)
    ).filter(lambda e: compact_name_key(e[0]) != compact_name_key(e[1]))
    unit = st.lists(edge, min_size=1, max_size=3, unique_by=lambda e: (e[0], e[1]))
    content = st.lists(unit, min_size=1, max_size=2)
    return st.fixed_dictionaries({p: st.none() | content for p in _PATHS}).map(
        lambda state: {path: c for path, c in state.items() if c}
    )


_spelled_corpora = st.sampled_from(sorted(_SPELLINGS)).flatmap(
    lambda lang: st.tuples(
        *(2 * [_corpus_of(["Depot", "Buyer", "Vendor", *_SPELLINGS[lang]])])
    )
)
_REMOVAL_POINTS = [
    "mid_removal",
    "after_removal",
    "in_merge",
    "after_commit_note",
    "mid_index_delta",
    "before_retired_deletes",
    "after_write_note",
    "before_record",
    "mid_commit",
]


@pytest.mark.slow
@pytest.mark.timeout(600)
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    corpora=_spelled_corpora,
    gone=st.sets(st.sampled_from(_PATHS), min_size=1, max_size=2),
    where=st.sampled_from(_WRITES),
    again=st.none() | st.sampled_from(_WRITES),
    point=st.none() | st.sampled_from(_REMOVAL_POINTS),
    k=st.integers(1, 3),
    salt=st.integers(0, 1000),
)
# Depot's vector write is rejected (its copy cites b.txt only); the removal
# of b.txt is killed after it stripped Depot's graph copy to a.txt's unit.
# The retry took the stale vector copy's units for Depot's, deleted it as
# vector-only with its edges, and a.txt's Depot -> Buyer was lost.
@example(
    corpora=(
        {"/corpus/b.txt": [[("Depot", "Acme Inc", 1), ("Buyer", "Depot", 2)]]},
        {
            "/corpus/a.txt": [[("Depot", "Buyer", 1)]],
            "/corpus/b.txt": [[("Depot", "Acme Inc", 1), ("Buyer", "Depot", 2)]],
        },
    ),
    gone={"/corpus/b.txt"},
    where="vector.upsert_entities",
    again=None,
    point="mid_index_delta",
    k=3,
    salt=0,
)
# 株式会社ひかり's graph write is rejected, the claim about it is stored.
# Without b.txt the types of ひかり and 株式会社ひかり agree and a full build
# has one entity: the commit folds the noted one into ひかり, and the stored
# claim follows only when it is derived again.
@example(
    corpora=(
        {},
        {
            "/corpus/b.txt": [[("Depot", "ひかり", 2)]],
            "/corpus/c.txt": [[("Depot", "株式会社ひかり", 3), ("Depot", "ひかり", 1)]],
            "/corpus/d.txt": [[("Depot", "ひかり", 2)]],
        },
    ),
    gone={"/corpus/b.txt"},
    where="graph.upsert_entities",
    again=None,
    point=None,
    k=1,
    salt=10,
)
def test_a_removal_after_rejected_writes_converges_when_interrupted(
    corpora: tuple[Corpus, Corpus],
    gone: set[str],
    where: str,
    again: str | None,
    point: str | None,
    k: int,
    salt: int,
) -> None:
    # A rejected run, then a removal of one or two of its documents
    # (interrupted at a point, or not; its writes rejected again, or not),
    # then clean runs.
    before, mid = corpora
    after = {path: content for path, content in mid.items() if path not in gone}
    config = Config()
    config.indexing.max_failure_rate = 0.95
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run(before)
    harness.run(mid, fail=_rejects(where, salt))
    harness.run(
        after,
        interrupt=point,
        k=k,
        fail=_rejects(again, salt + 1) if again else None,
    )
    harness.run(after)
    harness.run(after)

    assert harness.state() == expected_state(after, config)
    assert harness.registry_problems(after) == []
    entities = set(harness.graph.ids("entities"))
    for claim in harness.vector.data.get("claims", {}).values():
        assert {claim.subject_id, claim.object_id or claim.subject_id} <= entities
