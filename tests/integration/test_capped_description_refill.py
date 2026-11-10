# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stored description the graph cut is written from its vector copy (AWS-free).

The graph store cuts a description at ``indexing.neptune.property_max_length``;
only the vector copy holds the whole of it. A stored item read back from the
graph to be written again takes the vector copy's description
(``IndexingManager._with_full_descriptions``). An item derived from its
records again gets its description from them, so the cases of a derived item
drop the record of one unit it cites: the derivation then keeps the item as
stored. A stored edge rewritten for an endpoint's new name is not derived.
Without the refill the cut description reached the vector copy too.

Each case runs twice, the graph cap small and out of reach: the vector
copies must agree (the cap changes the graph copy only).
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.incremental_runs import (
    Corpus,
    IncrementalRun,
    document,
    extract,
)
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.integration

_NAMES = {
    "en": ("Vendor", "Buyer", "Depot"),
    "ko": ("공급사", "구매자", "창고"),
    "ja": ("仕入先", "買い手", "倉庫"),
}
_LANGS = pytest.mark.parametrize("names", list(_NAMES.values()), ids=list(_NAMES))


def _config(cap: int) -> Config:
    config = Config()
    config.indexing.max_failure_rate = 0.95
    config.indexing.neptune.property_max_length = cap
    # Unsummarized, a description over several units is over the cap.
    config.processing.graph_extraction.description_summarization.enabled = False
    return config


def _drop_record(harness: IncrementalRun, path: str, corpus: Corpus) -> None:
    """Delete the graph record of ``path``'s one unit (as a build of
    ``corpus`` writes it)."""
    doc = document(path, corpus[path]).document_id
    (unit,) = [u for u in extract(corpus).units if doc in (u.document_ids or [])]
    key = f"g-{unit.id}"
    dropped = [k for k in harness.vector.extraction_records if k[1] == key]
    assert dropped
    for k in dropped:
        del harness.vector.extraction_records[k]


def _vector_copies(
    scenario: Callable[[IncrementalRun], None], cap: int
) -> dict[str, object]:
    harness = IncrementalRun(FakeDocStatusStore(), _config(cap))
    scenario(harness)
    state = harness.state()
    return {
        key: state[key]
        for key in ("vector_entity_content", "vector_relationship_content")
    }


def _check(scenario: Callable[[IncrementalRun], None]) -> None:
    capped = _vector_copies(scenario, 40)
    assert capped == _vector_copies(scenario, 1_000_000)


@_LANGS
def test_a_strip_keeps_the_whole_description_of_an_item_it_cannot_derive(
    names: tuple[str, str, str],
) -> None:
    # The removal of b.txt strips Vendor, Buyer and Vendor -> Buyer, which
    # c.txt states too; c.txt's record is gone, so they keep their stored
    # descriptions, read back from the graph cut.
    v, b, _ = names
    corpus: Corpus = {f"/corpus/{p}.txt": [[(v, b, 2)]] for p in ("a", "b", "c")}

    def scenario(harness: IncrementalRun) -> None:
        harness.run(corpus)
        _drop_record(harness, "/corpus/c.txt", corpus)
        harness.run({p: c for p, c in corpus.items() if p != "/corpus/b.txt"})

    _check(scenario)


@_LANGS
def test_a_noted_edge_s_endpoints_keep_their_whole_descriptions(
    names: tuple[str, str, str],
) -> None:
    # The graph write of Vendor -> Buyer (d.txt states it again) is
    # rejected and noted; the next commit joins it with no delta, so its
    # endpoints are read back from the graph, cut, and c.txt's record is
    # gone.
    v, b, d = names
    corpus: Corpus = {f"/corpus/{p}.txt": [[(v, b, 2)]] for p in ("a", "b", "c")}
    more = {**corpus, "/corpus/d.txt": [[(v, b, 2), (d, v, 2)]]}
    (edge,) = [r for r in extract(more).relationships if r.target_name == b]

    def scenario(harness: IncrementalRun) -> None:
        harness.run(corpus)
        harness.run(
            more,
            fail=lambda at, item: at == "graph.upsert_relationships"
            and item == edge.id,
        )
        _drop_record(harness, "/corpus/c.txt", more)
        harness.run(more)

    _check(scenario)


@pytest.mark.parametrize(
    ("stored", "spelled"),
    [("Acme", "Acme Inc"), ("한빛", "주식회사 한빛"), ("ひかり", "株式会社ひかり")],
    ids=["en", "ko", "ja"],
)
def test_the_edges_of_a_noted_entity_keep_their_whole_descriptions(
    stored: str, spelled: str
) -> None:
    # The new documents' spelling re-keys the stored entity, and the
    # commit is killed after its note: the retry reads the noted entity's
    # stored edges and their other endpoint (Depot, not in the delta)
    # back from the graph, cut, and c.txt's record is gone.
    corpus: Corpus = {
        f"/corpus/{p}.txt": [[(stored, "Buyer", 2), (stored, "Depot", 2)]]
        for p in ("a", "b", "c")
    }
    more = {
        **corpus,
        **{f"/corpus/{p}.txt": [[(spelled, "Buyer", 2)]] for p in "defgh"},
    }

    def scenario(harness: IncrementalRun) -> None:
        harness.run(corpus)
        assert harness.run(more, interrupt="after_commit_note")
        _drop_record(harness, "/corpus/c.txt", more)
        harness.run(more)

    _check(scenario)


@pytest.mark.parametrize(
    ("stored", "renamed"),
    # One identity key each (case, quote marks, half-width kana).
    [("Acme", "ACME"), ("한빛", '"한빛"'), ("ヒカリ", "ﾋｶﾘ")],
    ids=["en", "ko", "ja"],
)
def test_a_stored_edge_renamed_keeps_its_whole_description(
    stored: str, renamed: str
) -> None:
    # The new documents' spelling renames the stored entity under its id:
    # its stored edge to Buyer, which no new document states, is written
    # again with the new name, its description read back from the graph.
    corpus: Corpus = {
        f"/corpus/{p}.txt": [[(stored, "Buyer", 2)]] for p in ("a", "b", "c")
    }
    more = {
        **corpus,
        **{f"/corpus/{p}.txt": [[(renamed, "Depot", 2)]] for p in "defgh"},
    }

    def scenario(harness: IncrementalRun) -> None:
        harness.run(corpus)
        harness.run(more)

    _check(scenario)
