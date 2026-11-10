# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A removal's strip whose write a store rejects is all or retry (AWS-free).

A strip writes in phases, each after the ones it depends on (claims under
new ids, the stored claims; new vertices, stored entities gaining a unit,
graph edges, vector edges, vector entities, kept entities). The first phase
with a rejected write stops it: what it did not write keeps its stored copy,
which still cites the removed text units or is in the pending-strip note, so
the retry derives it again. Each case below lost a kept document's item for
good, or left a stale one, while writes went on after a rejection.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.incremental_runs import (
    Corpus,
    IncrementalRun,
    expected_state,
    extract,
)
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.integration


def _config() -> Config:
    config = Config()
    config.indexing.max_failure_rate = 0.95
    config.indexing.max_document_attempts = 50
    return config


def _converges(harness: IncrementalRun, corpus: Corpus, config: Config) -> None:
    for _ in range(3):
        harness.run(corpus)
    assert harness.state() == expected_state(corpus, config)
    assert harness.registry_problems(corpus) == []


@pytest.mark.parametrize(
    "names",
    [
        ("Acme Inc", "Acme Corp", "Hikari KK", "Hanbit Co"),
        ("주식회사 한빛", "한빛 주식회사", "창고", "공급사"),
        ("株式会社ひかり", "ひかり株式会社", "倉庫", "仕入先"),
        ("Acme Inc", "Acme Corp", "ひかり株式会社", "(주)한빛"),
    ],
    ids=["en", "ko", "ja", "mix"],
)
def test_a_rejected_merge_target_keeps_the_edge_re_pointed_at_it(
    names: tuple[str, str, str, str],
) -> None:
    # Deleting g0.txt and g1.txt folds the first spelling into the second
    # and re-points k0.txt's edge at it. The graph write of the second is
    # rejected: written on, the edge sat at a vertex whose graph copy did
    # not cite k0.txt's unit, and deleting h0.txt then deleted both.
    a1, a2, x, y = names
    kept: Corpus = {"/corpus/k0.txt": [[("Buyer", a1, 1)]]}
    shared: Corpus = {**kept, "/corpus/h0.txt": [[(a2, x, 2)], [(a2, y, 1)]]}
    before: Corpus = {
        **shared,
        "/corpus/g0.txt": [[(x, a2, 2)]],
        "/corpus/g1.txt": [[("Buyer", y, 1)]],
    }
    target = next(e.id for e in extract(before).entities if e.name == a2)
    config = _config()
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run(before)
    rejected: list[str] = []

    def fail(at: str, item_id: str) -> bool:
        if at == "graph.upsert_entities" and item_id == target:
            rejected.append(item_id)
            return True
        return False

    harness.run(shared, fail=fail)

    assert rejected == [target]
    _converges(harness, kept, config)


@pytest.mark.parametrize(
    "names",
    [
        ("Acme", "Acme Inc", "Acme Corp", "Buyer"),
        ("한빛", "주식회사 한빛", "한빛 주식회사", "구매자"),
        ("ひかり", "株式会社ひかり", "ひかり株式会社", "買い手"),
        ("ひかり", "株式会社ひかり", "ひかり株式会社", "구매자"),
    ],
    ids=["en", "ko", "ja", "mix"],
)
def test_a_rejected_split_part_keeps_an_origin_another_document_shares(
    names: tuple[str, str, str, str],
) -> None:
    # Deleting g0.txt splits the short spelling off the entity the long
    # ones fold into, which h0.txt states too. The part's graph write is
    # rejected: the origin, which gains k0.txt's unit, was written anyway,
    # and its graph copy was the only one holding the part's lines; the
    # part and its edge never came back.
    short, long1, long2, other = names
    kept: Corpus = {
        "/corpus/k0.txt": [[("Vendor", short, 2)], [(long1, "Carrier", 1)]],
        "/corpus/h0.txt": [[(long2, "Depot", 2)], [(other, long2, 1)]],
    }
    before: Corpus = {**kept, "/corpus/g0.txt": [[(long2, "Depot", 2)]]}
    part = next(e.id for e in extract(kept).entities if e.name == short)
    config = _config()
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run(before)

    harness.run(kept, fail=lambda at, i: at == "graph.upsert_entities" and i == part)

    _converges(harness, kept, config)


@pytest.mark.parametrize(
    "spellings",
    [
        ("ACME", "Acme"),
        ("(주)한빛", "한빛"),
        ("(株)ひかり", "ひかり"),
        ("ACME", "ひかり"),
    ],
    ids=["en", "ko", "ja", "mix"],
)
def test_two_failed_removals_leave_no_stale_vector_edge(
    spellings: tuple[str, str],
) -> None:
    # Deleting g0.txt renames the entity (its graph edge writes rejected);
    # deleting h0.txt renames it back (its vector edge writes rejected).
    # Written on, the first left the entities' copies renamed, so the
    # retry saw no rename and the vector edge kept the first rename.
    final_name, other_name = spellings
    kept: Corpus = {"/corpus/k0.txt": [[(final_name, "Depot", 1)]]}
    shared: Corpus = {
        **kept,
        "/corpus/h0.txt": [[("Buyer", other_name, 1)], [("Vendor", other_name, 1)]],
    }
    config = _config()
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run({**shared, "/corpus/g0.txt": [[(final_name, "Depot", 1)]]})

    harness.run(shared, fail=lambda at, _i: at == "graph.upsert_relationships")
    harness.run(kept, fail=lambda at, _i: at == "vector.upsert_relationships")

    _converges(harness, kept, config)


@pytest.mark.parametrize(
    "spellings",
    [
        ("Acme Corp", "ACME Inc", "Acme"),
        ("한빛 주식회사", "(주)한빛", "한빛"),
        ("ひかり株式会社", "(株)ひかり", "ひかり"),
    ],
    ids=["en", "ko", "ja"],
)
def test_a_rejected_split_off_claim_keeps_the_claim_it_was_split_from(
    spellings: tuple[str, str, str],
) -> None:
    # n0.txt's claim folds into k0.txt's while h0.txt and h1.txt join the
    # spellings; deleting them splits it off under a new id. That write is
    # rejected: the stored claim, written stripped anyway, was the only copy
    # of n0.txt's statement, and no run derived it again.
    long_, paren, short = spellings
    kept: Corpus = {
        "/corpus/k0.txt": [[(long_, "Buyer", 3)]],
        "/corpus/n0.txt": [
            [("Vendor", long_, 1), ("Buyer", paren, 2), (short, "Buyer", 3)]
        ],
    }
    gone: Corpus = {
        "/corpus/h0.txt": [[(long_, "Depot", 1)]],
        "/corpus/h1.txt": [[("Depot", paren, 1)]],
    }
    config = _config()
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run({"/corpus/k0.txt": kept["/corpus/k0.txt"], **gone})
    harness.run({**kept, **gone})
    split_off = set(expected_state(kept, config)["claims"]) - set(
        harness.state()["claims"]
    )
    assert split_off

    harness.run(
        kept, fail=lambda at, i: at == "vector.upsert_claims" and i in split_off
    )

    _converges(harness, kept, config)
