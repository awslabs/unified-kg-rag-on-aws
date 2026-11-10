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
from collections.abc import Callable

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from tests.fixtures.doc_status_records import all_records
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.incremental_runs import (
    Content,
    Corpus,
    IncrementalRun,
    expected_state,
    extract,
)
from unified_kg_rag.domain.ingestion.graph_resolver import entity_keys
from unified_kg_rag.domain.ingestion.names import compact_name_key
from unified_kg_rag.domain.models import Config, DocStatus
from unified_kg_rag.shared.utils import entity_key

pytestmark = pytest.mark.property

Edge = tuple[str, str, int]

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


def _unit_of(names: list[str]) -> st.SearchStrategy[list[Edge]]:
    edge = st.tuples(
        st.sampled_from(names), st.sampled_from(names), st.integers(1, 3)
    ).filter(lambda e: compact_name_key(e[0]) != compact_name_key(e[1]))
    return st.lists(edge, min_size=1, max_size=3, unique_by=lambda e: (e[0], e[1]))


def _corpus_of(names: list[str]) -> st.SearchStrategy[Corpus]:
    content = st.lists(_unit_of(names), min_size=1, max_size=2)
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
# Vendor's graph write is rejected, so its edges are held back, but the
# vector copy of c.txt's Vendor -> Depot already has the rank d.txt's new
# Depot edge gave it. The removal of d.txt derives the rank back to the
# graph copy's: the edge was not written again, and its vector copy kept
# the rejected run's rank.
@example(
    corpora=(
        {
            "/corpus/c.txt": [[("Vendor", "ひかり", 2), ("Vendor", "Depot", 2)]],
            "/corpus/d.txt": [[("Vendor", "ひかり", 2), ("Vendor", "Depot", 2)]],
            "/corpus/a.txt": [
                [("Vendor", "ひかり株式会社", 1)],
                [
                    ("Vendor", "Buyer", 3),
                    ("Vendor", "ひかり", 3),
                    ("ひかり株式会社", "Vendor", 3),
                ],
            ],
        },
        {
            "/corpus/c.txt": [[("Vendor", "ひかり", 2), ("Vendor", "Depot", 2)]],
            "/corpus/d.txt": [
                [("Vendor", "ひかり", 1), ("株式会社ひかり", "Depot", 2)]
            ],
            "/corpus/a.txt": [
                [("Vendor", "ひかり株式会社", 1)],
                [
                    ("Vendor", "Buyer", 3),
                    ("Vendor", "ひかり", 3),
                    ("ひかり株式会社", "Vendor", 3),
                ],
            ],
        },
    ),
    gone={"/corpus/d.txt"},
    where="graph.upsert_entities",
    again=None,
    point=None,
    k=1,
    salt=2,
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


# The shapes of the bugs a removal after a rejected write had, drawn
# directly: the random corpora above did not reach them in 1,600 examples
# (outside their pinned ones), and the ci profile replays the same examples
# on every run.
_PLAIN = ["Depot", "Buyer", "Vendor"]


def _stating(draw: st.DrawFn, name: str, names: list[str], strength: int) -> Edge:
    """An edge stating ``name`` with another of ``names``, either way."""
    other = draw(
        st.sampled_from(
            [n for n in names if compact_name_key(n) != compact_name_key(name)]
        )
    )
    return (name, other, strength) if draw(st.booleans()) else (other, name, strength)


def _with_others(
    draw: st.DrawFn, names: list[str], documents: dict[str, Content]
) -> Corpus:
    """``documents`` and random ones over ``names`` at the other paths."""
    others = draw(_corpus_of(names))
    return {
        **{p: c for p, c in others.items() if p not in documents},
        **documents,
    }


@st.composite
def _shared_entity_removals(draw: st.DrawFn) -> tuple[Corpus, str, str]:
    """An entity a kept and a removed document both state, in one language's
    spellings or Korean and Japanese ones together: the rejected run's
    corpus, the removed path and the name."""
    lang = draw(st.sampled_from([*sorted(_SPELLINGS), "mix"]))
    spellings = _SPELLINGS.get(lang) or [*_SPELLINGS["ko"], *_SPELLINGS["ja"]]
    names = [*_PLAIN, *spellings]
    shared = draw(st.sampled_from(names))
    kept, gone = draw(st.permutations(_PATHS))[:2]
    documents = {
        path: [
            [_stating(draw, shared, names, draw(st.integers(1, 3)))],
            *draw(st.lists(_unit_of(names), max_size=1)),
        ]
        for path in (kept, gone)
    }
    return _with_others(draw, names, documents), gone, shared


@st.composite
def _folding_removals(draw: st.DrawFn) -> tuple[Corpus, str, str]:
    """Two spellings of one entity a full build keeps apart only while the
    removed document types the second one otherwise (an edge's strength
    parity types its names, and types are voted on): the kept document
    states the first one with a claim (strength 3) and the second one with
    the first one's type, the removed one states the second one in two
    units with the other type, so its removal folds them into one entity.
    The rejected run's corpus, the removed path and the first spelling."""
    lang = draw(st.sampled_from(sorted(_SPELLINGS)))
    first, second = draw(
        st.permutations(_SPELLINGS[lang])
        .map(lambda names: names[:2])
        .filter(lambda pair: entity_key(pair[0]) != entity_key(pair[1]))
    )
    kept, gone = draw(st.permutations(_PATHS))[:2]
    documents = {
        kept: [
            [
                _stating(draw, first, _PLAIN, 3),
                _stating(draw, second, _PLAIN, draw(st.sampled_from([1, 3]))),
            ]
        ],
        gone: [[_stating(draw, second, _PLAIN, 2)] for _ in range(2)],
    }
    return _with_others(draw, _PLAIN, documents), gone, first


def _entities_of(corpus: Corpus, name: str) -> set[str]:
    """The ids of the entities of ``name``'s spellings in a build of
    ``corpus``."""
    return {
        e.id
        for e in extract(corpus).entities
        if any(
            compact_name_key(key) == compact_name_key(name) for key in entity_keys(e)
        )
    }


def _converges(
    before: Corpus,
    mid: Corpus,
    gone: str,
    rejects: Callable[[str, str], bool],
    again: Callable[[str, str], bool] | None,
    point: str | None,
    k: int,
) -> None:
    """A rejected run over ``mid``, the removal of ``gone`` (interrupted at
    ``point``, or not; its writes rejected by ``again``, or not), then clean
    runs, leave what a full build stores, and no claim about an entity the
    graph does not hold."""
    after = {path: content for path, content in mid.items() if path != gone}
    config = Config()
    config.indexing.max_failure_rate = 0.95
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run(before)
    harness.run(mid, fail=rejects)
    harness.run(after, interrupt=point, k=k, fail=again)
    harness.run(after)
    harness.run(after)

    assert harness.state() == expected_state(after, config)
    assert harness.registry_problems(after) == []
    entities = set(harness.graph.ids("entities"))
    for claim in harness.vector.data.get("claims", {}).values():
        assert {claim.subject_id, claim.object_id or claim.subject_id} <= entities


@pytest.mark.slow
@pytest.mark.timeout(600)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    shape=_shared_entity_removals(),
    # The strip writes the graph copies first: the second to fifth writes
    # cover a kill between an entity's graph and vector copies.
    k=st.integers(2, 5),
    again=st.none() | st.sampled_from(_WRITES),
    salt=st.integers(0, 1000),
)
def test_a_removal_killed_in_its_strip_after_a_rejected_vector_write_converges(
    shape: tuple[Corpus, str, str],
    k: int,
    again: str | None,
    salt: int,
) -> None:
    # The removed document is indexed first; the entity's vector write is
    # rejected while the kept one is added, so its vector copy cites the
    # removed document only. The removal is killed at its k-th store write.
    mid, gone, shared = shape
    rejected = _entities_of(mid, shared)
    _converges(
        {gone: mid[gone]},
        mid,
        gone,
        lambda at, item_id: at == "vector.upsert_entities" and item_id in rejected,
        _rejects(again, salt) if again else None,
        "mid_index_delta",
        k,
    )


@pytest.mark.slow
@pytest.mark.timeout(600)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    shape=_folding_removals(),
    indexed_first=st.booleans(),
    where=st.sampled_from([w for w in _WRITES if "entit" in w]),
    point=st.none() | st.sampled_from(_REMOVAL_POINTS),
    k=st.integers(1, 3),
)
def test_a_removal_that_folds_a_rejected_spelling_converges(
    shape: tuple[Corpus, str, str],
    indexed_first: bool,
    where: str,
    point: str | None,
    k: int,
) -> None:
    # The write of the spelling the removal folds into the other is
    # rejected, the claim about it stored: the claim follows the fold only
    # when it is derived again.
    mid, gone, first = shape
    after = {path: content for path, content in mid.items() if path != gone}
    rejected = _entities_of(mid, first) - _entities_of(after, first)
    assert rejected
    _converges(
        {gone: mid[gone]} if indexed_first else {},
        mid,
        gone,
        lambda at, item_id: at == where and item_id in rejected,
        None,
        point,
        k,
    )
