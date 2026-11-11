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
    Killed,
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


@pytest.mark.parametrize("capped", [False, True], ids=["full", "capped"])
@pytest.mark.parametrize(
    "spellings",
    [
        ("ACME", "Acme"),
        ("한빛  물산", "한빛 물산"),
        ("ひかり  物産", "ひかり 物産"),
        ("ACME ひかり", "Acme ひかり"),
    ],
    ids=["en", "ko", "ja", "mix"],
)
def test_a_rename_and_back_leaves_no_stale_endpoint_name_on_a_vector_edge(
    spellings: tuple[str, str], capped: bool
) -> None:
    # Two spellings of one id. Deleting g0.txt names the entity after
    # h0.txt; its vector write is rejected, so its graph copy keeps the
    # first name while k0.txt's edge is renamed in both stores. Deleting
    # h0.txt and h1.txt names it back; its graph edge is written, its vector
    # edge rejected. The retry found the entity's graph copy under the name
    # it derives, saw no rename, and updated the edge's lineage only: its
    # vector copy kept the second name for good.
    first, second = spellings
    kept: Corpus = {"/corpus/k0.txt": [[(first, "Buyer", 1)]]}
    other: Corpus = {
        "/corpus/h0.txt": [[(second, "Vendor", 3)]],
        "/corpus/h1.txt": [[("Depot", second, 1)]],
    }
    before: Corpus = {**kept, "/corpus/g0.txt": [[(first, "Vendor", 3)]], **other}
    entity = next(e.id for e in extract(before).entities if e.name in spellings)
    config = _config()
    if capped:
        config.indexing.neptune.property_max_length = 40
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run(before)

    harness.run(
        {**kept, **other},
        fail=lambda at, i: at == "vector.upsert_entities" and i == entity,
    )
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


@pytest.mark.parametrize("stop", ["rejected", "killed"])
@pytest.mark.parametrize(
    "spellings",
    [
        ("ACME", "Acme Inc", "Acme Corp"),
        ("(주)한빛", "주식회사 한빛", "한빛 주식회사"),
        ("(株)ひかり", "株式会社ひかり", "ひかり株式会社"),
        ("(株)ひかり", "株式会社ひかり", "ひかり 주식회사"),
    ],
    ids=["en", "ko", "ja", "mix"],
)
def test_an_edge_under_a_new_id_stopped_before_its_vector_copy_is_derived_again(
    spellings: tuple[str, str, str], stop: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deleting g0.txt re-keys k0.txt's and k1.txt's edge; the strip writes
    # its graph copy, its description cut at the graph's limit, and stops
    # before the vector copy (rejected, or the run dies). No copy of the new
    # edge cites a removed unit: unnoted, the next strip rewrote it from the
    # graph copy, the cut description included.
    short, long1, long2 = spellings
    kept: Corpus = {
        f"/corpus/k{i}.txt": [[("Vendor", short, 3), ("Buyer", short, 2)]]
        for i in range(2)
    }
    shared: Corpus = {
        **kept,
        "/corpus/h0.txt": [[(short, "Vendor", 1), ("Buyer", long2, 2)]],
    }
    config = _config()
    config.indexing.neptune.property_max_length = 40
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run(
        {
            **shared,
            "/corpus/g0.txt": [
                [("Depot", long1, 2), (long2, "Depot", 1)],
                [(long2, "Buyer", 1)],
            ],
        }
    )

    if stop == "rejected":
        harness.run(shared, fail=lambda at, _i: at == "vector.upsert_relationships")
    else:

        def killed(*_args: object, **_kwargs: object) -> None:
            raise Killed("between the strip's graph and vector edge writes")

        with monkeypatch.context() as patched:
            patched.setattr(harness.vector, "upsert_relationships", killed)
            assert harness.run(shared)

    _converges(harness, kept, config)


@pytest.mark.parametrize("stop", ["rejected", "killed"])
@pytest.mark.parametrize(
    "names",
    [
        ("Acme", "ACME", "Acme Corp", "Acme Inc", "Depot", "Vendor", "Buyer"),
        (
            "한빛",
            "(주)한빛",
            "한빛 주식회사",
            "주식회사 한빛",
            "창고",
            "공급사",
            "구매자",
        ),
        (
            "ひかり",
            "(株)ひかり",
            "ひかり株式会社",
            "株式会社ひかり",
            "倉庫",
            "仕入先",
            "買い手",
        ),
        (
            "ひかり",
            "(株)ひかり",
            "ひかり株式会社",
            "株式会社ひかり",
            "창고",
            "공급사",
            "구매자",
        ),
    ],
    ids=["en", "ko", "ja", "mix"],
)
def test_a_retried_claim_re_key_leaves_no_claim_at_a_retired_entity(
    names: tuple[str, ...], stop: str
) -> None:
    # The commit adding n0.txt and n1.txt has the graph write of the short
    # spelling rejected (noted). Deleting h0.txt folds the "Corp" spelling
    # into it and re-keys n0.txt's claim about it; the strip stops after
    # deriving the claims (a later phase rejected, or killed at its first
    # write). The retry gave the claim derivation the folded entity's stored
    # copy, still stored, found by its name key; derived as itself, it held
    # the claim at its old id, beside the one the re-key gave it, about an
    # entity the run deleted.
    short, upper, corp, inc, depot, vendor, buyer = names
    kept: Corpus = {
        "/corpus/k0.txt": [[(depot, vendor, 3)]],
        "/corpus/k1.txt": [[(depot, inc, 1)]],
    }
    added: Corpus = {
        "/corpus/n0.txt": [
            [(buyer, corp, 3), (vendor, short, 3), (depot, short, 3)],
            [(short, vendor, 2)],
        ],
        "/corpus/n1.txt": [[(buyer, depot, 2), (depot, upper, 2), (short, buyer, 2)]],
    }
    gone: Corpus = {"/corpus/h0.txt": [[(corp, depot, 3)]]}
    final: Corpus = {
        **kept,
        **added,
        "/corpus/m0.txt": [[(inc, depot, 1), (buyer, short, 3)]],
    }
    config = _config()
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run(
        {
            **kept,
            **gone,
            "/corpus/g0.txt": [
                [(vendor, depot, 1), (upper, depot, 1), (corp, buyer, 3)]
            ],
        }
    )
    noted = {
        e.id for e in extract({**kept, **gone, **added}).entities if e.name == short
    }
    harness.run(
        {**kept, **gone, **added},
        fail=lambda at, i: at == "graph.upsert_entities" and i in noted,
    )

    if stop == "rejected":
        harness.run(final, fail=lambda at, _i: at == "vector.upsert_relationships")
    else:
        assert harness.run(final, interrupt="mid_index_delta")

    _converges(harness, final, config)
