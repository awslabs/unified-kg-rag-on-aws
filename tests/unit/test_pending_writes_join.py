# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which rejected writes a commit notes, and how a note meets the next delta.

``failed_writes`` decides what a note covers: a result naming an id no
written item has is not covered (what failed is not known), so its owners
are recorded FAILED instead of trusting a note that misses the item.
``PendingWrites.join`` folds the noted items into the next commit's delta:
a noted edge the delta states again takes the delta's endpoints, a noted
claim the delta states again keeps the noted statement, and a community or
report the delta makes again replaces the noted one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unified_kg_rag.application.storage.pending_writes import (
    NotedWrites,
    PendingWrites,
    failed_writes,
)
from unified_kg_rag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    Entity,
    Relationship,
)
from unified_kg_rag.ports.indexer import IndexingStats

pytestmark = pytest.mark.unit


def _rejected(*ids: str, total: int = 2) -> IndexingStats:
    stats = IndexingStats(total_items=total)
    stats.add_error("write rejected", len(ids), ids=list(ids))
    return stats


_VENDOR = Entity(id="ent-vendor", name="Vendor", text_unit_ids=["u1"])
_BUYER = Entity(id="ent-buyer", name="Buyer", text_unit_ids=["u2"])


def test_a_failed_id_no_written_item_has_leaves_its_result_uncovered() -> None:
    # One of the two failed ids is a written entity's, the other none's:
    # the note would miss an item, so the result is not covered at all.
    assert (
        failed_writes(
            {"vector_entities": _rejected(_VENDOR.id, "ent-unknown")}, [_VENDOR, _BUYER]
        )
        == {}
    )


def test_the_failed_ids_of_written_items_are_covered() -> None:
    covered = failed_writes(
        {
            "vector_entities": _rejected(_VENDOR.id),
            # Not a noted kind: a text unit's failure is its document's.
            "vector_text_units": _rejected("u1"),
        },
        [_VENDOR, _BUYER],
    )
    assert covered == {"vector_entities": [_VENDOR]}


def test_a_failure_naming_no_id_covers_every_written_item_of_its_kind() -> None:
    stats = IndexingStats(total_items=2)
    stats.add_error("store down", 2)
    assert failed_writes({"graph_entities": stats}, [_VENDOR, _BUYER]) == {
        "graph_entities": [_VENDOR, _BUYER]
    }


def _pending() -> PendingWrites:
    # join() reads the graph only for an endpoint neither the delta nor the
    # note holds; every endpoint below is held.
    return PendingWrites(SimpleNamespace(_stripped_ids={}))  # type: ignore[arg-type]


def _edge(rel_id: str, source: Entity, target: Entity, unit: str) -> Relationship:
    return Relationship(
        id=rel_id,
        source_id=source.id,
        target_id=target.id,
        source_name=source.name,
        target_name=target.name,
        type="RELATED_TO",
        description=f"{source.name} -> {target.name} in {unit}",
        text_unit_ids=[unit],
    )


@pytest.mark.parametrize(
    "names", [("Acme", "Acme Inc"), ("(주)한빛", "주식회사 한빛")], ids=["en", "ko"]
)
def test_a_noted_edge_the_delta_states_again_takes_the_deltas_endpoints(
    names: tuple[str, str],
) -> None:
    # The note names the subject by a spelling the delta no longer resolves
    # to; every noted edge at that endpoint follows the delta's resolution.
    old_name, new_name = names
    old = Entity(id="ent-old", name=old_name, text_unit_ids=["u1"])
    new = Entity(id="ent-new", name=new_name, text_unit_ids=["u1", "u3"])
    noted_edge = _edge("rel-1", old, _BUYER, "u1")
    other_edge = _edge("rel-2", old, _VENDOR, "u1")
    restated = _edge("rel-1", new, _BUYER, "u3")

    _, rels, *_ = _pending().join(
        NotedWrites(relationships=[noted_edge, other_edge]),
        [new, _BUYER, _VENDOR],
        [restated],
        None,
        None,
        None,
    )

    assert rels is not None
    by_id = {r.id: r for r in rels}
    assert {(r.source_id, r.source_name) for r in rels} == {(new.id, new_name)}
    assert by_id["rel-1"].text_unit_ids == ["u1", "u3"]
    assert by_id["rel-2"].target_id == _VENDOR.id


def test_a_noted_claim_the_delta_states_again_keeps_the_noted_statement() -> None:
    def claim(unit: str, status: str) -> Claim:
        return Claim(
            id="claim-1",
            subject_id=_VENDOR.id,
            subject_name=_VENDOR.name,
            object_id=_BUYER.id,
            object_name=_BUYER.name,
            type="RATES",
            status=status,
            description=f"Vendor rates Buyer in {unit}",
            text_unit_ids=[unit],
        )

    _, _, claims, _, _ = _pending().join(
        NotedWrites(claims=[claim("u1", "TRUE")]),
        None,
        None,
        [claim("u2", "FALSE")],
        None,
        None,
    )

    assert claims is not None
    (joined,) = claims
    assert joined.text_unit_ids == ["u1", "u2"]
    assert joined.description == "Vendor rates Buyer in u1\nVendor rates Buyer in u2"
    assert joined.status == "disputed"


def test_a_community_and_report_the_delta_makes_again_replace_the_noted_ones() -> None:
    def community(name: str, units: list[str]) -> Community:
        return Community(
            id="com-1",
            name=name,
            level="0",
            parent="-1",
            children=[],
            text_unit_ids=units,
        )

    def report(title: str) -> CommunityReport:
        return CommunityReport(
            id="rep-1", name=title, community_id="com-1", summary=title
        )

    kept = Community(
        id="com-2",
        name="kept",
        level="0",
        parent="-1",
        children=[],
        text_unit_ids=["u9"],
    )

    *_, communities, reports = _pending().join(
        NotedWrites(
            communities=[community("noted", ["u1"]), kept],
            community_reports=[report("noted")],
        ),
        None,
        None,
        None,
        [community("remade", ["u2"])],
        [report("remade")],
    )

    assert communities is not None and reports is not None
    assert [(c.id, c.name, c.text_unit_ids) for c in communities] == [
        ("com-1", "remade", ["u2"]),
        ("com-2", "kept", ["u9"]),
    ]
    assert [r.summary for r in reports] == ["remade"]
