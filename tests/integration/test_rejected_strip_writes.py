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


def test_a_rejected_strip_write_is_logged_with_the_item_ids(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A rejection that depends on the item (its size, a mapping conflict)
    # blocks the namespace until it is written or the namespace is reset:
    # the WARNING names the items, not only how many.
    kept: Corpus = {"/corpus/k0.txt": [[("Buyer", "Acme Inc", 1)]]}
    shared: Corpus = {**kept, "/corpus/h0.txt": [[("Acme Corp", "Depot", 2)]]}
    before: Corpus = {**shared, "/corpus/g0.txt": [[("Depot", "Acme Corp", 2)]]}
    depot = next(e.id for e in extract(before).entities if e.name == "Depot")
    harness = IncrementalRun(FakeDocStatusStore(), _config())
    harness.run(before)

    with caplog.at_level("WARNING"):
        harness.run(
            shared, fail=lambda at, i: at == "graph.upsert_entities" and i == depot
        )

    assert any(
        record.levelname == "WARNING" and depot in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "names",
    [
        ("Acme Co", "Acme Corp", "Depot", "Vendor", "Buyer"),
        ("주식회사 한빛", "한빛 주식회사", "창고", "공급사", "구매자"),
        ("ひかり", "株式会社ひかり", "倉庫", "仕入先", "買い手"),
        ("ひかり", "株式会社ひかり", "창고", "공급사", "구매자"),
    ],
    ids=["en", "ko", "ja", "mix"],
)
def test_a_rejected_split_part_leaves_its_neighbours_rank(
    names: tuple[str, str, str, str, str],
) -> None:
    # Deleting g0.txt and g1.txt splits the second spelling off the first;
    # the part's new vertex is rejected. The derivation also gives the
    # stored neighbour both spellings link (not read back by the strip),
    # ranked with the part's edge: taken for a new vertex, it was written
    # with the rejected part, and kept that rank once the retry (h0.txt
    # deleted too) folded the spellings back.
    first, second, depot, vendor, buyer = names
    kept: Corpus = {"/corpus/k0.txt": [[(depot, first, 2)], [(depot, second, 2)]]}
    other: Corpus = {"/corpus/h0.txt": [[(vendor, first, 1)]]}
    config = _config()
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run(
        {
            **kept,
            **other,
            "/corpus/g0.txt": [[(vendor, buyer, 1)]],
            "/corpus/g1.txt": [[(second, vendor, 1)]],
        }
    )
    part = {e.id for e in extract({**kept, **other}).entities if e.name == second}
    rejected: list[str] = []

    def fail(at: str, item_id: str) -> bool:
        if at == "graph.upsert_entities" and item_id in part:
            rejected.append(item_id)
            return True
        return False

    harness.run({**kept, **other}, fail=fail)

    assert rejected
    _converges(harness, kept, config)


@pytest.mark.parametrize(
    "names",
    [
        ("“Acme”", "Acme Inc", "Acme Corp", "Buyer"),
        ("「한빛」", "한빛 주식회사", "주식회사 「한빛」", "구매자"),
        ("「ひかり」", "ひかり・株式会社", "株式会社「ひかり」", "買い手"),
        ("「ひかり」", "ひかり・株式会社", "株式会社「ひかり」", "구매자"),
    ],
    ids=["en", "ko", "ja", "mix"],
)
@pytest.mark.parametrize("stop", ["rejected", "killed"])
def test_a_stored_edge_that_gains_a_statement_of_its_unit_is_noted(
    names: tuple[str, str, str, str], stop: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # k1.txt's first unit states the edge to the buyer under two spellings.
    # Deleting i0.txt moves the first spelling's statement from the third
    # spelling's edge to the second's, which cites that unit already: its
    # graph copy gains a statement but no unit, and its vector write is
    # rejected (or the run dies before it). No copy cites a removed unit,
    # so the retry did not find it: its vector copy lacked the line and
    # the strength for good.
    first, second, third, buyer = names
    final: Corpus = {
        "/corpus/k0.txt": [[(buyer, first, 2), (buyer, second, 1)]],
        "/corpus/k1.txt": [
            [(second, buyer, 3), (first, buyer, 3)],
            [(third, buyer, 2)],
        ],
        "/corpus/n0.txt": [[(third, buyer, 2)]],
    }
    config = _config()
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run({**final, "/corpus/i0.txt": [[(buyer, first, 2)]]})

    if stop == "rejected":
        harness.run(final, fail=lambda at, _i: at == "vector.upsert_relationships")
    else:

        def killed(*_args: object, **_kwargs: object) -> None:
            raise Killed("between the strip's graph and vector edge writes")

        with monkeypatch.context() as patched:
            patched.setattr(harness.vector, "upsert_relationships", killed)
            patched.setattr(harness.vector, "update_relationship_lineage", killed)
            assert harness.run(final)

    _converges(harness, final, config)


@pytest.mark.parametrize(
    "names",
    [
        ("Acme", "Acme Co", "ACME Inc.", "Acme Corp", "Buyer", "Depot", "Vendor"),
        (
            "한빛",
            "주식회사 한빛",
            "(주)한빛",
            "한빛 주식회사",
            "구매자",
            "창고",
            "공급사",
        ),
        (
            "株式会社ひかり",
            "ひかり",
            "ひかり株式会社",
            "(株)ひかり",
            "買い手",
            "倉庫",
            "仕入先",
        ),
        (
            "株式会社ひかり",
            "ひかり",
            "ひかり株式会社",
            "(株)ひかり",
            "구매자",
            "창고",
            "공급사",
        ),
    ],
    ids=["en", "ko", "ja", "mix"],
)
@pytest.mark.parametrize("stop", ["rejected", "killed"])
def test_a_stored_edge_that_gains_a_unit_is_noted_before_its_vector_copy(
    names: tuple[str, ...], stop: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Deleting i0.txt changes which spelling the short name resolves to,
    # and k0.txt's edge from the buyer merges into a stored edge of
    # m0.txt's under its id: its graph copy gains k0.txt's unit, and its
    # vector write is rejected (or the run dies before it). No copy of that
    # edge cites a removed unit, so the retry did not find it: its vector
    # copy lacked k0.txt's line for good.
    short, long1, paren, long2, buyer, depot, vendor = names
    final: Corpus = {
        "/corpus/k0.txt": [
            [(short, buyer, 2)],
            [(buyer, short, 1), (depot, long1, 1), (paren, depot, 1)],
        ],
        "/corpus/k1.txt": [[(paren, depot, 2)]],
        "/corpus/n0.txt": [[(long2, vendor, 1)]],
        "/corpus/m0.txt": [[(short, depot, 2), (buyer, long2, 3)], [(buyer, paren, 2)]],
    }
    config = _config()
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run({**final, "/corpus/i0.txt": [[(short, depot, 3)]]})

    if stop == "rejected":
        harness.run(final, fail=lambda at, _i: at == "vector.upsert_relationships")
    else:

        def killed(*_args: object, **_kwargs: object) -> None:
            raise Killed("between the strip's graph and vector edge writes")

        with monkeypatch.context() as patched:
            patched.setattr(harness.vector, "upsert_relationships", killed)
            patched.setattr(harness.vector, "update_relationship_lineage", killed)
            assert harness.run(final)

    _converges(harness, final, config)


@pytest.mark.parametrize(
    "names",
    [
        ("Acme", "Acme Inc", "Acme Corp", "Buyer", "Vendor"),
        ("한빛", "주식회사 한빛", "(주)한빛", "구매자", "공급사"),
        ("ひかり", "株式会社ひかり", "(株)ひかり", "買い手", "仕入先"),
        ("ひかり", "株式会社ひかり", "(株)ひかり", "구매자", "공급사"),
    ],
    ids=["en", "ko", "ja", "mix"],
)
def test_a_rejected_entity_gaining_a_name_key_keeps_the_spelling_it_takes(
    names: tuple[str, str, str, str, str],
) -> None:
    # Deleting g0.txt and g1.txt moves the third spelling (stated in
    # k0.txt's unit, which the second already cites) from the first
    # spelling's entity to the second's: the second gains its name key, not
    # a unit, and its graph write is rejected. Written on, the first lost
    # the key; deleting h0.txt then deleted it, and the derivation, finding
    # no stored entity of the third spelling's key, dropped k0.txt's edge
    # from it.
    first, second, third, buyer, vendor = names
    kept: Corpus = {"/corpus/k0.txt": [[(second, buyer, 1), (third, vendor, 2)]]}
    other: Corpus = {
        "/corpus/h0.txt": [
            [(third, buyer, 1), (buyer, second, 1)],
            [(first, vendor, 2), (buyer, second, 3)],
        ]
    }
    before: Corpus = {
        **kept,
        **other,
        "/corpus/g0.txt": [[(buyer, third, 2), (first, vendor, 3)]],
        "/corpus/g1.txt": [
            [(first, buyer, 2)],
            [(vendor, second, 3), (first, vendor, 2)],
        ],
    }
    taker = {e.id for e in extract(before).entities if e.name == second}
    config = _config()
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run(before)

    harness.run(
        {**kept, **other},
        fail=lambda at, i: at == "graph.upsert_entities" and i in taker,
    )

    _converges(harness, kept, config)


_STRANDED_NAMES = pytest.mark.parametrize(
    "names",
    [
        ("Acme Inc", "Acme", "Acme Corp", "ACME", "Vendor", "Buyer"),
        ("(주)한빛", "한빛", "한빛 주식회사", "한빛 ", "공급사", "구매자"),
        ("ひかり株式会社", "ひかり", "株式会社ひかり", "ひかり", "仕入先", "買い手"),
        ("Acme Inc", "Acme", "Acme 주식회사", "ACME", "仕入先", "구매자"),
    ],
    ids=["en", "ko", "ja", "mix"],
)


def _strand_an_edge(
    names: tuple[str, str, str, str, str, str],
    stop: str,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[IncrementalRun, Config, Corpus, Corpus]:
    """A harness whose last run stopped a strip before re-pointing a kept
    document's edge, and the kept and the removed corpus.

    Deleting h0.txt splits the first spelling off the entity the others
    fold into (the type vote changes): the strip writes the part and the
    origin, which now cites only i0.txt's units, and stops before
    re-pointing k0.txt's edge at the part (rejected, or the run dies)."""
    first, plain, other, upper, vendor, buyer = names
    kept: Corpus = {"/corpus/k0.txt": [[(first, vendor, 1)]]}
    removed: Corpus = {"/corpus/h0.txt": [[(buyer, plain, 1)]]}
    shared: Corpus = {
        **kept,
        "/corpus/i0.txt": [[("Depot", other, 2)], [(vendor, upper, 2)]],
    }
    config = _config()
    harness = IncrementalRun(FakeDocStatusStore(), config)
    harness.run({**shared, **removed})

    if stop == "rejected":
        harness.run(shared, fail=lambda at, _i: at == "graph.upsert_relationships")
    else:

        def killed(*_args: object, **_kwargs: object) -> None:
            raise Killed("between the strip's vertex and edge writes")

        with monkeypatch.context() as patched:
            patched.setattr(harness.graph, "upsert_relationships", killed)
            assert harness.run(shared)
    return harness, config, kept, removed


@pytest.mark.parametrize("returned", [False, True], ids=["removed", "returned"])
@pytest.mark.parametrize("stop", ["rejected", "killed"])
@_STRANDED_NAMES
def test_an_edge_left_at_a_vertex_citing_only_removed_units_is_kept(
    names: tuple[str, str, str, str, str, str],
    stop: str,
    returned: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The next run deletes i0.txt (with h0.txt, or with h0.txt back):
    # deleting the origin, every unit of it removed, deleted k0.txt's edge
    # with it, and k0.txt lost its only edge for good.
    harness, config, kept, removed = _strand_an_edge(names, stop, monkeypatch)

    _converges(harness, {**kept, **removed} if returned else kept, config)


@pytest.mark.parametrize("after", ["unnoted", "killed_before_retiring"])
@_STRANDED_NAMES
def test_a_vertex_holding_a_kept_edge_goes_once_the_edge_is_re_pointed(
    names: tuple[str, str, str, str, str, str],
    after: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The removal finds the edge at the vertex in the graph, not only in
    # the pending-strip note; and the vertex, left citing no text unit, is
    # not written but retired: a run killed before the retired ids are
    # deleted leaves its stored copy, which still cites the removed units.
    harness, config, kept, _removed = _strand_an_edge(names, "rejected", monkeypatch)
    if after == "unnoted":
        harness.vector.delete_extraction_records(["pending-strip"], suffix="default")
        harness.run(kept)
    else:
        deletes: list[object] = []
        delete = harness.graph.delete_by_id

        def killed_second(*args: object, **kwargs: object) -> object:
            deletes.append(args)
            if len(deletes) == 2:
                raise Killed("before the strip's retired ids are deleted")
            return delete(*args, **kwargs)

        with monkeypatch.context() as patched:
            patched.setattr(harness.graph, "delete_by_id", killed_second)
            assert harness.run(kept)
        # No copy without provenance: the vertex keeps the units a retry
        # finds it by.
        graph_entities = harness.graph.read_entities(
            sorted(harness.graph.ids("entities"))
        )
        assert all(e.text_unit_ids for e in graph_entities)

    _converges(harness, kept, config)
