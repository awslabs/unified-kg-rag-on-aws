# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A community report's findings cite the records they rest on.

The report prompt numbers the entities and relationships it packs and asks,
as MS GraphRAG's does (``prompts/index/community_report.py``), for evidence
only and a ``[Data: Entities (...); Relationships (...)]`` reference per
finding. The references leave the prose and are kept, as record ids, on the
finding, in every report language.
"""

from __future__ import annotations

from typing import Any

import pytest

from unified_kg_rag.adapters.ingestion.community_detector import CommunityDetector
from unified_kg_rag.domain.ingestion.graph_builder import GraphBuilder
from unified_kg_rag.domain.ingestion.report_references import (
    REPORT_FINDINGS_FIELD,
    report_cited_ids,
    report_cited_records,
    split_data_references,
)
from unified_kg_rag.domain.models import Community, Config, Entity, Relationship
from unified_kg_rag.domain.prompts import CommunityReportPrompt

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("text", "prose", "cited"),
    [
        (
            "Vendor supplies Buyer [Data: Entities (1, 3); Relationships (2)].",
            "Vendor supplies Buyer.",
            {"entities": [1, 3], "relationships": [2]},
        ),
        (
            "Vendor ships late [Data: Entities (1, 2, 3, 4, 5, +more), "
            "Claims (7)] and pays [data: relationships (4, 4)]",
            "Vendor ships late and pays",
            {"entities": [1, 2, 3, 4, 5], "relationships": [4]},
        ),
        (
            "벤더는 구매자에게 부품을 공급한다 [Data: Entities (2); Relationships (1)]",
            "벤더는 구매자에게 부품을 공급한다",
            {"entities": [2], "relationships": [1]},
        ),
        # Full-width punctuation a model writing Japanese or Chinese may use.
        (
            "供給者は買い手に部品を納入する［Data：Entities（1，2）；Relationships（3）］。",
            "供給者は買い手に部品を納入する。",
            {"entities": [1, 2], "relationships": [3]},
        ),
        ("供应商向买方交付零件。", "供应商向买方交付零件。", {}),
    ],
)
def test_references_are_split_from_the_prose(
    text: str, prose: str, cited: dict[str, list[int]]
) -> None:
    assert split_data_references(text) == (prose, cited)


def _detector() -> CommunityDetector:
    config = Config()
    config.graph.community_detection.report_generation.enabled = False
    detector = CommunityDetector(config, show_progress=False)
    detector.graph = GraphBuilder(
        [
            Entity(id="v", name="벤더", type="ORG", description="부품 공급사"),
            Entity(id="b", name="Buyer", type="ORG", description="Buys parts"),
            Entity(id="p", name="部品", type="PRODUCT", description="Parts"),
        ],
        [
            Relationship(id="r-vb", source_id="v", target_id="b", type="SUPPLIES"),
            Relationship(id="r-vp", source_id="v", target_id="p", type="MAKES"),
        ],
        [],
    ).build()
    return detector


def _community() -> Community:
    return Community(
        id="c0",
        name="Level 0 Community c0",
        level="0",
        parent="",
        children=[],
        entity_ids=["v", "b", "p"],
        text_unit_ids=[],
        size=3,
    )


def test_the_packed_input_is_numbered_in_packing_order() -> None:
    detector = _detector()
    report_input = detector._prepare_report_input(_community())
    (chain_input,) = detector._create_report_chain_inputs([report_input])
    entity_lines = chain_input["entities"].splitlines()
    relationship_lines = chain_input["relationships"].splitlines()
    assert [line.split("]")[0] for line in entity_lines] == ["- [1", "- [2", "- [3"]
    assert [line.split("]")[0] for line in relationship_lines] == ["- [1", "- [2"]
    for number, entity in enumerate(report_input["entities"], 1):
        assert entity_lines[number - 1].startswith(f"- [{number}] {entity['name']} (")
    # The relationship's id is what a reference resolves to; it is not in
    # the prompt.
    assert {r["id"] for r in report_input["relationships"]} == {"r-vb", "r-vp"}
    assert "r-vb" not in chain_input["relationships"]


def test_findings_keep_the_records_they_cite() -> None:
    detector = _detector()
    community = _community()
    report_input = detector._prepare_report_input(community)
    entities = [e["id"] for e in report_input["entities"]]
    relationships = [r["id"] for r in report_input["relationships"]]
    result: dict[str, Any] = {
        "community_name": "벤더 공급망",
        "summary": "벤더가 구매자에게 부품을 공급한다.",
        "rating": "6",
        "rating_explanation": "공급 관계가 중심이다.",
        "findings": {
            "finding": [
                {
                    "summary": "벤더가 중심이다",
                    "explanation": "벤더는 구매자와 부품을 잇는다 "
                    "[Data: Entities (1, 2, 9); Relationships (2, 1, +more)].",
                },
                {
                    "summary": "部品の供給 [Data: Entities (3)]",
                    "explanation": "部品は供給される［Data：Relationships（2）］。",
                },
                {"summary": "근거 없는 요약", "explanation": "참조가 없다."},
            ]
        },
    }
    report = detector._create_community_report(
        community, result, {}, report_input=report_input
    )
    first, second, third = report.findings
    # Number 9 is not in the input: dropped.
    assert first.entity_ids == entities[:2]
    assert first.relationship_ids == [relationships[1], relationships[0]]
    assert first.explanation == "벤더는 구매자와 부품을 잇는다."
    assert second.summary == "部品の供給"
    assert second.entity_ids == [entities[2]]
    assert second.relationship_ids == [relationships[1]]
    assert second.explanation == "部品は供給される。"
    assert (third.entity_ids, third.relationship_ids) == ([], [])
    # The input's numbers mean nothing outside it: not in the searched text.
    assert "Data" not in report.full_content


def test_the_prompt_asks_for_evidence_and_references_only() -> None:
    system = CommunityReportPrompt.system_prompt_template
    human = CommunityReportPrompt.human_prompt_template
    assert (
        "Do not include information where the supporting evidence for it is not "
        "provided" in system
    )
    assert "[Data: Entities (1, 3); Relationships (2, 7)]" in system
    assert "+more" in system
    # The reference syntax stays ASCII whatever the report language.
    assert "[Data: ...] references keep their exact ASCII form" in human
    for speculative in ("strategic", "actionable", "quantified", "opportunities"):
        assert speculative not in (system + human).lower()


def test_a_stored_reports_cited_ids_are_read_from_its_findings() -> None:
    metadata = {
        REPORT_FINDINGS_FIELD: [
            {"summary": "a", "entity_ids": ["e1", "e2"], "relationship_ids": ["r1"]},
            {"summary": "b", "entity_ids": ["e2", "e3", ""], "relationship_ids": []},
            "not a finding",
            {"summary": "c", "entity_ids": "e9", "relationship_ids": ["r2", "r1"]},
        ]
    }
    assert report_cited_ids(metadata) == (["e1", "e2", "e3"], ["r1", "r2"])


@pytest.mark.parametrize(
    "metadata", [{}, {REPORT_FINDINGS_FIELD: None}, {REPORT_FINDINGS_FIELD: "x"}]
)
def test_a_report_stored_without_findings_cites_nothing(
    metadata: dict[str, Any],
) -> None:
    assert report_cited_ids(metadata) == ([], [])
    assert report_cited_records(metadata) == []


def test_a_reports_cited_records_come_finding_by_finding() -> None:
    # Global search and DRIFT cite a report by these records' text units:
    # each finding's relationships (the narrower evidence) before its
    # entities, findings in order, each record once.
    metadata = {
        REPORT_FINDINGS_FIELD: [
            {"entity_ids": ["e1", "e2"], "relationship_ids": ["r1"]},
            "not a finding",
            {"entity_ids": ["e2", "e3", ""], "relationship_ids": ["r2", "r1", 7]},
        ]
    }
    assert report_cited_records(metadata) == [
        ("relationships", "r1"),
        ("entities", "e1"),
        ("entities", "e2"),
        ("relationships", "r2"),
        ("entities", "e3"),
    ]


@pytest.mark.parametrize(
    ("text", "prose", "cited"),
    [
        # MS GraphRAG's parent reports cite their sub-reports.
        (
            "Vendor leads the region [Data: Reports (2, 1); Entities (3)].",
            "Vendor leads the region.",
            {"reports": [2, 1], "entities": [3]},
        ),
        # A range reads as every number in it.
        (
            "벤더가 공급한다 [Data: Entities (1-4); Relationships (2 – 3)]",
            "벤더가 공급한다",
            {"entities": [1, 2, 3, 4], "relationships": [2, 3]},
        ),
        # The lenticular brackets a model writing Japanese or Chinese uses.
        (
            "供給者が納入する【Data：Reports（1）；Entities（2～3）】。",
            "供給者が納入する。",
            {"reports": [1], "entities": [2, 3]},
        ),
        (
            "供应商交付零件【Data: Relationships (1)】",
            "供应商交付零件",
            {"relationships": [1]},
        ),
    ],
)
def test_reports_ranges_and_lenticular_brackets_are_read(
    text: str, prose: str, cited: dict[str, list[int]]
) -> None:
    assert split_data_references(text) == (prose, cited)


def test_a_range_reads_only_the_numbers_the_input_has() -> None:
    _, cited = split_data_references(
        "x [Data: Entities (2-1000000000)]", sizes={"entities": 3}
    )
    assert cited == {"entities": [2, 3]}


def _rollup_parent() -> tuple[CommunityDetector, Community, dict[str, Any]]:
    """A parent over the entity cap with two reported children, its input."""
    from unified_kg_rag.domain.models import CommunityFinding, CommunityReport

    detector = _detector()
    detector.community_detection_config.report_generation.max_entities_per_report = 1
    parent = _community().model_copy(
        update={"id": "p", "level": "1", "children": ["k1", "k2"]}
    )

    def child(cid: str, size: int, entity: str, relationship: str) -> Any:
        return CommunityReport(
            id=f"r-{cid}",
            name=f"Child {cid}",
            community_id=cid,
            summary=f"{cid} summary",
            size=size,
            findings=[
                CommunityFinding(
                    summary="s", entity_ids=[entity], relationship_ids=[relationship]
                )
            ],
        )

    reports = {"k1": child("k1", 5, "b", "r-vb"), "k2": child("k2", 2, "p", "r-vp")}
    return detector, parent, detector._prepare_report_input(parent, reports)


def test_sub_reports_are_numbered_and_cited_by_their_findings_records() -> None:
    detector, parent, report_input = _rollup_parent()
    (chain_input,) = detector._create_report_chain_inputs([report_input])
    lines = chain_input["sub_community_reports"].splitlines()
    # Numbered in packing order (biggest first), as the entities are.
    assert [line.split("] ")[0] for line in lines] == ["- [1", "- [2"]
    assert "Child k1" in lines[0] and "Child k2" in lines[1]
    result: dict[str, Any] = {
        "community_name": "공급망",
        "summary": "요약",
        "rating": "5",
        "findings": {
            "finding": [
                {
                    "summary": "하위 커뮤니티가 공급을 맡는다",
                    "explanation": "두 하위 보고서가 그렇다 "
                    "[Data: Reports (2, 1, 7); Entities (1)].",
                },
                {
                    "summary": "部品",
                    "explanation": "部品は供給される【Data: Reports (1)】。",
                },
            ]
        },
    }
    report = detector._create_community_report(
        parent, result, {}, report_input=report_input
    )
    first, second = report.findings
    own = report_input["entities"][0]["id"]
    # The parent's own record first, then each cited child's records (7 is no
    # sub-report: dropped).
    assert first.entity_ids == [own, *[e for e in ("p", "b") if e != own]]
    assert first.relationship_ids == ["r-vp", "r-vb"]
    assert (second.entity_ids, second.relationship_ids) == (["b"], ["r-vb"])
    assert second.explanation == "部品は供給される。"


def test_the_prompt_lets_a_finding_cite_its_sub_reports() -> None:
    system = CommunityReportPrompt.system_prompt_template
    assert "Reports (" in system
