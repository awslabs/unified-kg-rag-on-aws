# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The extraction store: unit-independent records, their keys and the index.

A text unit's record is stored without what ties it to the unit (its id and
placement attributes) and re-stamped onto any unit with the same text; the
key follows the settings that shape the record, built-in prompt templates
included; the OpenSearch extraction-records index (here on the embedded
engine) is upsert-only and outside the artifact indices a reset clears.
"""

from __future__ import annotations

import pytest

from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.stores import FakeVectorStore
from unified_kg_rag.adapters.local.search_engine import SQLiteSearchEngine
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.application.ingestion.extraction_store import ExtractionStore
from unified_kg_rag.domain.ingestion.contributions import (
    claims_from_payload,
    claims_payload,
    record_from_payload,
    record_key,
    record_payload,
)
from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
)
from unified_kg_rag.domain.models import (
    Claim,
    Config,
    Entity,
    LanguageCode,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.domain.prompts import GraphExtractionPrompt
from unified_kg_rag.ports.indexer import IndexingStats
from unified_kg_rag.shared.utils.cache_keys import unit_record_fingerprint

pytestmark = pytest.mark.unit

TEXT = "Vendor ships widgets to Buyer."


def _unit(unit_id: str, suffix: str | None = None, region: str = "north") -> TextUnit:
    attributes = {"filters": {"region": region}, "file_name": f"{unit_id}.txt"}
    if suffix:
        attributes["index"] = suffix
    return TextUnit(id=unit_id, text=TEXT, attributes=attributes)


def _record(unit: TextUnit) -> TextUnitExtraction:
    placement = {k: v for k, v in unit.attributes.items() if k in ("index", "filters")}
    edge = Relationship(
        id="r",
        source_id="v",
        target_id="b",
        source_name="Vendor",
        target_name="Buyer",
        type="SHIPS_TO",
        attributes={**placement, "evidence": "kept"},
    )
    apply_text_unit_weights(edge, {unit.id: 4.0})
    return TextUnitExtraction(
        text_unit_id=unit.id,
        entities=[
            Entity(
                id="v",
                name="Vendor",
                type="ORG",
                confidence=0.8,
                text_unit_ids=[unit.id],
                attributes=dict(placement),
                name_embedding=[0.1],
            )
        ],
        relationships=[edge],
        rejected={"phantom": "ungrounded"},
        gleaned=True,
    )


class TestPayload:
    def test_a_record_is_stored_without_its_unit_and_restamped(self) -> None:
        payload = record_payload(_record(_unit("t1", "tenant-a", "north")))
        assert "t1" not in str(payload) and "north" not in str(payload)

        other = _unit("t2", "tenant-b", "south")
        record = record_from_payload(payload, other)

        assert (record.text_unit_id, record.stored, record.gleaned) == (
            "t2",
            True,
            True,
        )
        assert record.rejected == {"phantom": "ungrounded"}
        (vendor,) = record.entities
        assert (vendor.name, vendor.confidence, vendor.text_unit_ids) == (
            "Vendor",
            0.8,
            ["t2"],
        )
        assert vendor.attributes == {
            "filters": {"region": "south"},
            "index": "tenant-b",
        }
        assert vendor.name_embedding is None
        (edge,) = record.relationships
        assert (edge.text_unit_ids, edge.weight) == (["t2"], 4.0)
        assert edge.attributes == {
            "evidence": "kept",
            "filters": {"region": "south"},
            "index": "tenant-b",
            "text_unit_weights": [4.0],
        }

    def test_claims_are_stored_without_their_unit(self) -> None:
        claim = Claim(
            id="c",
            subject_id="",
            subject_name="Vendor",
            object_name="Buyer",
            type="PAYMENT",
            text_unit_ids=["t1"],
            attributes={"filters": {"region": "north"}},
        )
        payload = claims_payload([claim])
        assert "t1" not in str(payload)
        (restored,) = claims_from_payload(payload, _unit("t2", region="south"))
        assert restored.text_unit_ids == ["t2"]
        assert restored.attributes == {"filters": {"region": "south"}}
        assert (restored.subject_name, restored.type) == ("Vendor", "PAYMENT")


class TestFingerprint:
    def test_follows_what_shapes_a_record(self) -> None:
        base = Config()
        fingerprint = unit_record_fingerprint(base, gleaned=True)
        assert fingerprint == unit_record_fingerprint(Config(), gleaned=True)
        assert fingerprint != unit_record_fingerprint(base, gleaned=False)
        assert fingerprint != unit_record_fingerprint(base, gleaned=True, claims=True)

        model = Config()
        model.models.roles.graph_extraction = "other-model"
        assert unit_record_fingerprint(model, gleaned=True) != fingerprint
        language = Config()
        language.processing.translation.target_language = LanguageCode.KO
        assert unit_record_fingerprint(language, gleaned=True) != fingerprint

    def test_ignores_the_settings_that_only_shape_the_text(self) -> None:
        # The record is keyed by the text too, so a chunking change re-keys
        # only the chunks whose text it changes.
        chunking = Config()
        chunking.processing.chunking.fallback_chunk_size = 77
        assert unit_record_fingerprint(chunking, gleaned=True) == (
            unit_record_fingerprint(Config(), gleaned=True)
        )

    def test_follows_the_built_in_prompt_templates(self, monkeypatch) -> None:
        before = unit_record_fingerprint(Config(), gleaned=False)
        monkeypatch.setattr(
            GraphExtractionPrompt,
            "system_prompt_template",
            GraphExtractionPrompt.system_prompt_template + "\nOne more rule.",
        )
        assert unit_record_fingerprint(Config(), gleaned=False) != before


class _Broken(FakeVectorStore):
    def read_extraction_records(self, keys, suffix=None):
        raise RuntimeError("store down")

    def write_extraction_records(self, records, suffix=None):
        raise RuntimeError("store down")


def _raw(payload: dict, unit: TextUnit) -> dict:
    return payload


class TestExtractionStore:
    def test_round_trip_by_text_and_suffix(self) -> None:
        vectors = FakeVectorStore()
        store = ExtractionStore(Config(), lambda: vectors)
        written = store.save([(_unit("t1", "a"), {"p": 1})], "fp")
        assert written == 1

        # Same text under the same suffix: found; another suffix: not.
        assert store.load([_unit("t9", "a"), _unit("t8", "b")], "fp", _raw) == {
            "t9": {"p": 1}
        }
        assert store.load([_unit("t9", "a")], "other-fp", _raw) == {}
        assert list(vectors.extraction_records) == [("a", record_key("fp", TEXT))]

    def test_without_reuse_nothing_is_read_and_records_are_refreshed(self) -> None:
        vectors = FakeVectorStore()
        ExtractionStore(Config(), lambda: vectors).save([(_unit("t1"), {"p": 1})], "fp")
        store = ExtractionStore(Config(), lambda: vectors, reuse=False)
        assert store.load([_unit("t1")], "fp", _raw) == {}
        assert store.save([(_unit("t1"), {"p": 2})], "fp") == 1
        assert ExtractionStore(Config(), lambda: vectors).load(
            [_unit("t1")], "fp", _raw
        ) == {"t1": {"p": 2}}

    def test_an_unreadable_record_is_extracted_again(self, caplog) -> None:
        vectors = FakeVectorStore()
        store = ExtractionStore(Config(), lambda: vectors)
        store.save([(_unit("t1"), {"entities": [{"id": "only-an-id"}]})], "fp")
        assert store.load([_unit("t1")], "fp", record_from_payload) == {}
        assert "unreadable extraction record" in caplog.text

    def test_an_unavailable_store_extracts_everything(self) -> None:
        def unavailable():
            raise RuntimeError("no endpoint")

        store = ExtractionStore(Config(), unavailable)
        assert store.load([_unit("t1")], "fp", _raw) == {}
        assert store.save([(_unit("t1"), {"p": 1})], "fp") == 0

    def test_a_failing_store_is_skipped(self, caplog) -> None:
        store = ExtractionStore(Config(), _Broken)
        assert store.load([_unit("t1")], "fp", _raw) == {}
        assert store.save([(_unit("t1"), {"p": 1})], "fp") == 0
        assert "store down" in caplog.text


def _indexer(path) -> OpenSearchIndexer:
    return OpenSearchIndexer(
        Config(),
        embedding_factory=HashingEmbeddingFactory(dimensions=8),
        opensearch_client=SQLiteSearchEngine(path),
    )


class TestOpenSearchExtractionRecords:
    def test_records_are_upserted_and_read_per_suffix(self, tmp_path) -> None:
        indexer = _indexer(tmp_path / "vector.sqlite3")
        stats = indexer.write_extraction_records({"k1": {"a": [1]}}, suffix="x")
        assert (stats.total_items, stats.successful_items) == (1, 1)
        indexer.write_extraction_records({"k1": {"a": [2]}, "k2": {}}, suffix="x")

        assert indexer.read_extraction_records(["k1", "k2", "k3"], suffix="x") == {
            "k1": {"a": [2]},
            "k2": {},
        }
        assert indexer.read_extraction_records(["k1"], suffix="y") == {}

    def test_a_reset_leaves_the_records(self, tmp_path) -> None:
        indexer = _indexer(tmp_path / "vector.sqlite3")
        indexer.write_extraction_records({"k1": {"a": 1}}, suffix="x")
        indexer.index_text_units([_unit("t1", "x")])

        indexer.clear(["x"])
        indexer.check_index_settings("x")

        assert indexer.read_text_unit_ids(["t1"], suffix="x") == []
        assert indexer.read_extraction_records(["k1"], suffix="x") == {"k1": {"a": 1}}

    def test_a_failed_write_is_reported(self, tmp_path, mocker) -> None:
        indexer = _indexer(tmp_path / "vector.sqlite3")
        mocker.patch.object(
            indexer.opensearch_client, "bulk_index", side_effect=RuntimeError("full")
        )
        stats = indexer.write_extraction_records({"k1": {}}, suffix="x")
        assert isinstance(stats, IndexingStats)
        assert (stats.failed_items, stats.successful_items) == (1, 0)
