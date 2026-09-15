# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Community-report source lineage (AWS-free).

A community report was the only indexed artifact with no pointer back to the
text units and documents it summarizes, so a retrieved report could not be cited
or filtered by origin. These tests follow the lineage the whole way: community
detection attaches it, the OpenSearch indexer writes it, and the retriever hands
it back on the parsed hit.
"""

from __future__ import annotations

import pytest

import unified_kg_rag.adapters.ingestion.community_detector as detector_module
from unified_kg_rag.adapters.ingestion.community_detector import CommunityDetector
from unified_kg_rag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import (
    Community,
    CommunityReport,
    Config,
    TextUnit,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def indexer(config: Config) -> OpenSearchIndexer:
    inst = OpenSearchIndexer.__new__(OpenSearchIndexer)
    inst.config = config
    inst.opensearch_config = config.indexing.opensearch
    inst.analyzer = "standard"
    inst.target_language = config.processing.translation.target_language.value
    inst._embedding_dimension = 1024
    return inst


@pytest.fixture
def retriever(config: Config) -> OpenSearchRetriever:
    inst = OpenSearchRetriever.__new__(OpenSearchRetriever)
    object.__setattr__(inst, "_config", config)
    object.__setattr__(inst, "_opensearch_config", config.indexing.opensearch)
    object.__setattr__(inst, "_field_mappings", inst._initialize_field_mappings())
    return inst


def _community(community_id: str, text_unit_ids: list[str]) -> Community:
    return Community(
        id=community_id,
        name=f"Community {community_id}",
        level="0",
        parent="",
        children=[],
        entity_ids=["e1"],
        relationship_ids=[],
        text_unit_ids=text_unit_ids,
    )


def _report(community_id: str, **kwargs) -> CommunityReport:
    return CommunityReport(
        id=f"report-{community_id}",
        name=f"Report for {community_id}",
        community_id=community_id,
        summary="A vendor supplies a buyer.",
        full_content="A vendor supplies a buyer.",
        **kwargs,
    )


class TestAttachReportLineage:
    def test_report_inherits_its_community_text_units(self) -> None:
        report = _report("c1")
        CommunityDetector._attach_report_lineage(
            [report], [_community("c1", ["t1", "t2"])], text_units=None
        )
        assert report.text_unit_ids == ["t1", "t2"]

    def test_documents_resolved_from_the_contributing_text_units(self) -> None:
        report = _report("c1")
        units = [
            TextUnit(id="t1", text="a", document_ids=["d1"]),
            TextUnit(id="t2", text="b", document_ids=["d2", "d1"]),
            # Not a member of this community, so its document must not appear.
            TextUnit(id="t9", text="c", document_ids=["d9"]),
        ]
        CommunityDetector._attach_report_lineage(
            [report], [_community("c1", ["t1", "t2"])], text_units=units
        )
        assert report.document_ids == ["d1", "d2"]

    def test_documents_empty_without_text_units(self) -> None:
        # Callers that do not supply the corpus still get text-unit lineage.
        report = _report("c1")
        CommunityDetector._attach_report_lineage(
            [report], [_community("c1", ["t1"])], text_units=None
        )
        assert report.text_unit_ids == ["t1"]
        assert report.document_ids == []

    def test_each_report_gets_its_own_community_lineage(self) -> None:
        first, second = _report("c1"), _report("c2")
        units = [
            TextUnit(id="t1", text="a", document_ids=["d1"]),
            TextUnit(id="t2", text="b", document_ids=["d2"]),
        ]
        CommunityDetector._attach_report_lineage(
            [first, second],
            [_community("c1", ["t1"]), _community("c2", ["t2"])],
            text_units=units,
        )
        assert (first.text_unit_ids, first.document_ids) == (["t1"], ["d1"])
        assert (second.text_unit_ids, second.document_ids) == (["t2"], ["d2"])

    def test_report_without_a_matching_community_gets_empty_lineage(self) -> None:
        report = _report("orphan")
        CommunityDetector._attach_report_lineage(
            [report], [_community("c1", ["t1"])], text_units=None
        )
        assert report.text_unit_ids == []
        assert report.document_ids == []


class TestGenerateReportsAttachesLineage:
    def test_generated_report_carries_lineage(self, config: Config, mocker) -> None:
        detector = CommunityDetector.__new__(CommunityDetector)
        detector.config = config
        detector.community_detection_config = config.graph.community_detection
        detector.community_detection_config.report_generation.enabled = True
        detector.community_detection_config.report_generation.enable_sub_community_rollup = (
            False
        )
        detector.ignore_errors = False
        detector.show_progress = False
        detector.graph = None
        detector.report_generator = mocker.Mock()
        detector.batch_processor = mocker.Mock()
        detector.batch_processor.execute_with_fallback.return_value = [
            {"community_name": "Supply", "summary": "A vendor supplies a buyer."}
        ]
        mocker.patch.object(
            CommunityDetector, "_extract_attributes_from_graph", return_value={}
        )
        mocker.patch.object(CommunityDetector, "_prepare_report_input", return_value={})
        mocker.patch.object(detector_module, "estimate_token_count", return_value=1)

        reports = detector.generate_reports(
            [_community("c1", ["t1"])],
            text_units=[TextUnit(id="t1", text="a", document_ids=["d1"])],
        )

        assert len(reports) == 1
        assert reports[0].text_unit_ids == ["t1"]
        assert reports[0].document_ids == ["d1"]


class TestLineageRoundTrip:
    def test_mapping_declares_both_lineage_fields_as_keyword(self, indexer) -> None:
        properties = indexer._get_community_reports_mapping()["mappings"]["properties"]
        assert properties["text_unit_ids"]["type"] == "keyword"
        assert properties["document_ids"]["type"] == "keyword"

    def test_indexed_document_carries_lineage(self, indexer) -> None:
        report = _report("c1", text_unit_ids=["t1", "t2"], document_ids=["d1"])
        doc = indexer._prepare_community_report_doc(report, ([0.1], [0.2], [0.3]))
        assert doc["text_unit_ids"] == ["t1", "t2"]
        assert doc["document_ids"] == ["d1"]

    def test_missing_lineage_indexes_as_an_empty_list(self, indexer) -> None:
        # A report from before these fields existed must still index cleanly.
        doc = indexer._prepare_community_report_doc(
            _report("c1"), ([0.1], [0.2], [0.3])
        )
        assert doc["text_unit_ids"] == []
        assert doc["document_ids"] == []

    def test_lineage_survives_index_then_retrieve(self, indexer, retriever) -> None:
        report = _report("c1", text_unit_ids=["t1", "t2"], document_ids=["d1", "d2"])
        doc = indexer._prepare_community_report_doc(report, ([0.1], [0.2], [0.3]))

        prefix = retriever._opensearch_config.community_reports_index_prefix
        result = retriever._parse_hit(
            {"_index": f"{prefix}-default", "_score": 1.0, "_source": doc}
        )

        assert result.metadata["text_unit_ids"] == ["t1", "t2"]
        assert result.metadata["document_ids"] == ["d1", "d2"]
