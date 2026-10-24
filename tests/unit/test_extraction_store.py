# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The extraction store: unit-independent records, their keys and the index.

A text unit's record is stored without what ties it to the unit (its id and
placement attributes) and re-stamped onto any unit with the same text; the
key follows the settings that shape the record, built-in prompt templates
included; the OpenSearch extraction-records index (here on the embedded
engine) is outside the artifact indices a reset clears, and keeps only the
records a full build names when asked to.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.stores import FakeVectorStore
from unified_kg_rag.adapters.local.search_engine import SQLiteSearchEngine
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.application.ingestion import extraction_store
from unified_kg_rag.application.ingestion.extraction_store import (
    ExtractionStore,
    with_record_keys,
)
from unified_kg_rag.domain.ingestion.contributions import (
    claims_from_payload,
    claims_payload,
    record_from_payload,
    record_key,
    record_keys,
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
from unified_kg_rag.domain.prompts import GraphExtractionPrompt, GraphRefinementPrompt
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

    def test_follows_the_gleaning_settings_of_a_gleaned_record_only(self) -> None:
        base = Config()
        rounds = Config()
        rounds.processing.gleaning.max_rounds += 1
        assert unit_record_fingerprint(rounds, gleaned=True) != (
            unit_record_fingerprint(base, gleaned=True)
        )
        assert unit_record_fingerprint(rounds, gleaned=False) == (
            unit_record_fingerprint(base, gleaned=False)
        )
        refinement = Config(
            custom_prompts={
                "graph_refinement_system": GraphRefinementPrompt.system_prompt_template
                + "\nOne more rule."
            }
        )
        assert unit_record_fingerprint(refinement, gleaned=True) != (
            unit_record_fingerprint(base, gleaned=True)
        )
        assert unit_record_fingerprint(refinement, gleaned=False) == (
            unit_record_fingerprint(base, gleaned=False)
        )

    def test_follows_the_built_in_refinement_prompt(self, monkeypatch) -> None:
        gleaned = unit_record_fingerprint(Config(), gleaned=True)
        plain = unit_record_fingerprint(Config(), gleaned=False)
        monkeypatch.setattr(
            GraphRefinementPrompt,
            "system_prompt_template",
            GraphRefinementPrompt.system_prompt_template + "\nOne more rule.",
        )
        assert unit_record_fingerprint(Config(), gleaned=True) != gleaned
        assert unit_record_fingerprint(Config(), gleaned=False) == plain

    def test_follows_the_claim_settings_of_a_claims_record_only(self) -> None:
        base = Config()
        base.processing.claim_extraction.enabled = True
        wider = base.model_copy(deep=True)
        wider.processing.claim_extraction.max_entities_per_prompt += 1
        assert unit_record_fingerprint(wider, gleaned=False, claims=True) != (
            unit_record_fingerprint(base, gleaned=False, claims=True)
        )
        assert unit_record_fingerprint(wider, gleaned=False) == (
            unit_record_fingerprint(base, gleaned=False)
        )

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
        assert store.save([(_unit("t1", "a"), {"p": 1})], "fp") == set()

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
        assert store.save([(_unit("t1"), {"p": 2})], "fp") == set()
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
        # Nothing stored: the caller fails the unit (derivation needs it).
        assert store.save([(_unit("t1"), {"p": 1})], "fp") == {"t1"}

    def test_a_failing_store_is_skipped(self, caplog) -> None:
        store = ExtractionStore(Config(), _Broken)
        assert store.load([_unit("t1")], "fp", _raw) == {}
        assert store.save([(_unit("t1"), {"p": 1})], "fp") == {"t1"}
        assert "store down" in caplog.text

    @pytest.mark.parametrize("reuse", [True, False])
    def test_a_refused_store_fails_before_extraction(self, reuse: bool) -> None:
        # A store the backend refuses to create (index settings it does not
        # know) failed every record write after the extraction; ``load``,
        # which runs before it, now raises. Another failure leaves the store
        # to its reads and writes, as before.
        from unified_kg_rag.shared import ConfigurationError
        from unified_kg_rag.shared.exceptions import StoreRequestRejectedError

        vectors = FakeVectorStore()
        prepared: list[str | None] = []

        def refuse(suffix=None):
            prepared.append(suffix)
            raise StoreRequestRejectedError("unknown setting [index.max_result_windw]")

        vectors.prepare_extraction_records = refuse  # type: ignore[method-assign]
        store = ExtractionStore(Config(), lambda: vectors, reuse=reuse)
        with pytest.raises(ConfigurationError, match="max_result_windw"):
            store.load([_unit("t1", "a")], "fp", _raw)
        assert prepared == ["a"]

        def unreachable(suffix=None):
            raise RuntimeError("store down")

        vectors.prepare_extraction_records = unreachable  # type: ignore[method-assign]
        assert store.load([_unit("t1", "a")], "fp", _raw) == {}

    def test_an_unreachable_store_fails_before_extraction(self) -> None:
        # It was logged as a failing cache: the run paid for the whole
        # extraction and then failed to write it.
        from unified_kg_rag.shared.exceptions import StoreConnectionError

        vectors = FakeVectorStore()

        def unreachable(suffix=None):
            raise StoreConnectionError("Failed to connect to OpenSearch.")

        vectors.prepare_extraction_records = unreachable  # type: ignore[method-assign]
        with pytest.raises(StoreConnectionError):
            ExtractionStore(Config(), lambda: vectors).load([_unit("t1")], "fp", _raw)

    def test_a_lost_lease_is_raised_by_the_preparation(self) -> None:
        # It was logged as a failing cache and the run went on extracting.
        from unified_kg_rag.shared import NamespaceLeaseError

        vectors = FakeVectorStore()

        def lost(suffix=None):
            raise NamespaceLeaseError("another run took the namespace")

        vectors.prepare_extraction_records = lost  # type: ignore[method-assign]
        with pytest.raises(NamespaceLeaseError):
            ExtractionStore(Config(), lambda: vectors).load([_unit("t1")], "fp", _raw)

    def test_save_returns_the_units_of_the_rejected_keys(self) -> None:
        vectors = FakeVectorStore()
        other = TextUnit(id="t2", text="Buyer pays Vendor.")
        rejected = record_key("fp", other.text)

        def partial(records, suffix=None):
            stats = IndexingStats(total_items=len(records))
            stats.add_error("rejected", ids=[rejected])
            stats.successful_items = len(records) - 1
            return stats

        vectors.write_extraction_records = partial  # type: ignore[method-assign]
        store = ExtractionStore(Config(), lambda: vectors)
        assert store.save([(_unit("t1"), {}), (other, {})], "fp") == {"t2"}

        # A failure no key is reported for fails every unit of the write.
        def unattributed(records, suffix=None):
            return IndexingStats(total_items=len(records), failed_items=1)

        vectors.write_extraction_records = unattributed  # type: ignore[method-assign]
        assert store.save([(_unit("t1"), {}), (other, {})], "fp") == {"t1", "t2"}


class TestRecordKeys:
    """Units name the keys of their stored records, computed, not stamped."""

    def test_keys_follow_the_record_and_its_failures(self) -> None:
        config = Config()
        units = [_unit(f"t{i}") for i in range(4)]
        units[0].attributes["extraction_records"] = {"graph": "stale"}
        records = [
            TextUnitExtraction(text_unit_id="t0"),
            TextUnitExtraction(text_unit_id="t1", gleaned=True),
            TextUnitExtraction(text_unit_id="t2"),
        ]
        keyed = with_record_keys(
            config,
            units,
            records,
            graph_failed={"t2"},
            claims=True,
            claims_failed={"t1"},
        )
        plain = unit_record_fingerprint(config, gleaned=False)
        gleaned = unit_record_fingerprint(config, gleaned=True)
        assert [record_keys(u) for u in keyed] == [
            {
                "graph": record_key(plain, TEXT),
                "claims": record_key(
                    unit_record_fingerprint(config, gleaned=False, claims=True), TEXT
                ),
            },
            {"graph": record_key(gleaned, TEXT)},
            {},  # its record was not stored
            {},  # no record
        ]
        # The units themselves are left as they are (resumable caches).
        assert units[0].attributes["extraction_records"] == {"graph": "stale"}
        assert "extraction_records" not in keyed[2].attributes
        assert keyed[1].attributes["filters"] == {"region": "north"}

    def test_the_fingerprint_is_computed_once_per_kind(self, mocker) -> None:
        # Regression: it was computed for every unit (~1.5 ms each), the bulk
        # of the indexing stage of a large build.
        spy = mocker.spy(extraction_store, "unit_record_fingerprint")
        units = [_unit(f"t{i}") for i in range(50)]
        records = [
            TextUnitExtraction(text_unit_id=u.id, gleaned=i % 2 == 0)
            for i, u in enumerate(units)
        ]

        keyed = with_record_keys(Config(), units, records, graph_failed=(), claims=True)

        assert spy.call_count == 4  # (gleaned or not) x (graph, claims)
        assert record_keys(keyed[0])["claims"] == record_key(
            unit_record_fingerprint(Config(), gleaned=True, claims=True), TEXT
        )

    def test_no_claims_key_when_claims_were_not_extracted(self) -> None:
        (unit,) = with_record_keys(
            Config(),
            [_unit("t0")],
            [TextUnitExtraction(text_unit_id="t0")],
            graph_failed=(),
            claims=False,
        )
        assert set(record_keys(unit)) == {"graph"}


def _indexer(path) -> OpenSearchIndexer:
    return OpenSearchIndexer(
        Config(),
        embedding_factory=HashingEmbeddingFactory(dimensions=8),
        opensearch_client=SQLiteSearchEngine(path),
    )


def _strict_engine(path, refused):
    """An embedded engine that refuses, as OpenSearch would, every index
    whose body ``refused`` names a reason for; and the names it was asked
    to create."""
    from unified_kg_rag.shared.exceptions import StoreRequestRejectedError

    engine = SQLiteSearchEngine(path)
    create = engine.create_index
    created: list[str] = []

    def strict(index, body, *, exist_ok=True):
        created.append(index)
        if reason := refused(body):
            raise StoreRequestRejectedError(reason, reason="illegal_argument_exception")
        return create(index, body, exist_ok=exist_ok)

    engine.create_index = strict  # type: ignore[method-assign]
    return engine, created


def _probing_indexer(config: Config, engine) -> OpenSearchIndexer:
    return OpenSearchIndexer(
        config,
        embedding_factory=HashingEmbeddingFactory(dimensions=8),
        opensearch_client=engine,
    )


class TestOpenSearchExtractionRecords:
    def test_preparing_creates_the_index_once_and_reports_a_refusal(
        self, tmp_path
    ) -> None:
        from unified_kg_rag.shared import ConfigurationError
        from unified_kg_rag.shared.exceptions import StoreRequestRejectedError

        indexer = _indexer(tmp_path / "vector.sqlite3")
        indexer.prepare_extraction_records("x")
        indexer.prepare_extraction_records("x")
        alias = indexer._get_name(
            indexer.opensearch_config.extraction_records_index_prefix, "x"
        )
        assert len(indexer.opensearch_client.get_indices_by_alias(alias)) == 1

        def refuse(index, body, *, exist_ok=True):
            raise StoreRequestRejectedError(
                "unknown setting [index.max_result_windw]",
                reason="illegal_argument_exception",
            )

        indexer.opensearch_client.create_index = refuse  # type: ignore[method-assign]
        with pytest.raises(ConfigurationError, match="index_settings"):
            indexer.prepare_extraction_records("y")

    @pytest.mark.parametrize("typo", ["refresh_intervl", "knnx"])
    def test_a_refused_setting_fails_also_when_the_index_exists(
        self, tmp_path, typo: str
    ) -> None:
        # Only a missing records index was created, so a reset or an
        # incremental run over an existing namespace extracted every unit
        # before the indexing stage's first index creation was refused. The
        # run creates and deletes a probe index with the vector indices'
        # settings (a "knn"-prefixed typo included) first.
        from unified_kg_rag.shared import ConfigurationError

        path = tmp_path / "vector.sqlite3"
        _indexer(path).prepare_extraction_records("x")  # the namespace exists
        config = Config()
        config.indexing.opensearch.index_settings = {
            **config.indexing.opensearch.index_settings,
            typo: "1s",
        }
        engine, created = _strict_engine(
            path, lambda body: typo in body["settings"] and f"unknown [{typo}]"
        )
        indexer = _probing_indexer(config, engine)
        with pytest.raises(ConfigurationError, match=typo) as refused:
            indexer.check_index_creation("x")
        # It names the index the refused probe stood in for (the probe's own
        # name is the extraction records'): the first one the run creates.
        target = indexer._get_name(indexer._specs()[0].alias_prefix, "x")
        assert f"(a probe of '{target}')" in str(refused.value)

        # Accepted settings: a probe per vector index is created and
        # deleted, the records index is left as it was.
        config.indexing.opensearch.index_settings.pop(typo)
        created.clear()
        indexer.check_index_creation("x")
        probes = [name for name in created if "settings-probe" in name]
        assert len(probes) == len(indexer._specs())
        assert not engine.get_aliases_by_index("*settings-probe*")

    def test_the_probe_holds_every_field_an_engine_may_refuse(self, tmp_path) -> None:
        # The probe index had no fields: a vector method or an analyzer the
        # engine refuses passed it and failed the first real index creation
        # after extraction.
        config = Config()
        config.processing.translation.enabled = True
        config.processing.translation.source_language = LanguageCode.KO
        config.processing.translation.target_language = LanguageCode.EN
        engine, created = _strict_engine(tmp_path / "vector.sqlite3", lambda _: None)
        bodies: list[dict] = []
        create = engine.create_index

        def keep(index, body, *, exist_ok=True):
            bodies.append(body)
            return create(index, body, exist_ok=exist_ok)

        engine.create_index = keep  # type: ignore[method-assign]
        indexer = _probing_indexer(config, engine)
        indexer.check_index_creation("x")

        # Each index's own mapping (one merged mapping kept only one
        # definition of a field several indices map), the records' too.
        assert bodies == [
            *(spec.mapping() for spec in indexer._specs()),
            indexer._get_extraction_records_mapping(),
        ]
        assert {"text", "translated_text_en"} <= set(
            bodies[0]["mappings"]["properties"]
        )

    def test_the_probes_cover_only_the_indices_the_run_creates(self, tmp_path) -> None:
        # One probe index was created for every run: an incremental run into
        # existing indices (it creates none) was refused at a cluster's shard
        # cap, and a full build with room for one index passed, then failed
        # in the indexing stage after extraction.
        from unified_kg_rag.shared.exceptions import StoreRequestRejectedError

        engine, created = _strict_engine(tmp_path / "vector.sqlite3", lambda _: None)
        config = Config()
        config.processing.claim_extraction.enabled = True
        indexer = _probing_indexer(config, engine)
        indexer.upsert_text_units([_unit("t1", "x")])
        indexer.prepare_extraction_records("x")
        built = set(engine.get_aliases_by_index("*"))
        room = {"indices": 0}
        create = engine.create_index

        def at_cap(index, body, *, exist_ok=True):
            if len(engine.get_aliases_by_index("*")) >= room["indices"]:
                raise StoreRequestRejectedError(
                    "validation_exception: this action would add [1] total "
                    "shards, but this cluster currently has [9]/[9] maximum "
                    "shards open;",
                    reason="validation_exception",
                )
            return create(index, body, exist_ok=exist_ok)

        engine.create_index = at_cap  # type: ignore[method-assign]

        def live() -> int:
            return len(engine.get_aliases_by_index("*"))

        # Incremental: only the vector indices without a live one, together.
        missing = [
            spec
            for spec in indexer._specs()
            if not engine.get_index_name_by_alias(
                indexer._get_name(spec.alias_prefix, "x")
            )
        ]
        assert 1 < len(missing) < len(indexer._specs())
        room["indices"] = live() + len(missing) - 1
        with pytest.raises(StoreRequestRejectedError, match="max_shards_per_node"):
            indexer.check_index_creation("x", rebuild=False)
        room["indices"] += 1
        created.clear()
        indexer.check_index_creation("x", rebuild=False)
        assert len(created) == len(missing)

        # Every alias live: an incremental run creates nothing to refuse.
        for spec in missing:
            alias = indexer._get_name(spec.alias_prefix, "x")
            index = indexer._new_index_name(alias, spec.alias_prefix, "x")
            create(index, spec.mapping())
            engine.update_alias(alias, index)
        room["indices"] = live()
        created.clear()
        indexer.check_index_creation("x", rebuild=False)
        assert created == []
        # Nor for a type it does not extract (its index is created only on a
        # full build).
        claims = indexer._get_name(indexer.opensearch_config.claims_index_prefix, "x")
        engine.delete_indices(engine.get_indices_by_alias(claims))
        config.processing.claim_extraction.enabled = False
        indexer.check_index_creation("x", rebuild=False)
        assert created == []

        # A full build or reset needs room for all its new indices at once.
        room["indices"] = live() + len(indexer._specs()) - 1
        with pytest.raises(StoreRequestRejectedError, match="max_shards_per_node"):
            indexer.check_index_creation("x")
        room["indices"] += 1
        indexer.check_index_creation("x")
        assert not engine.get_aliases_by_index("*settings-probe*")
        assert set(engine.get_aliases_by_index("*")) >= built

    def test_the_sweep_leaves_a_sibling_namespaces_indices(self, tmp_path) -> None:
        # A namespace whose suffix extends '<suffix>-settings-probe-' lost its
        # live extraction-records index to the other namespace's sweep.
        engine, _ = _strict_engine(tmp_path / "vector.sqlite3", lambda _: None)
        indexer = _probing_indexer(Config(), engine)
        indexer.prepare_extraction_records("a-settings-probe-x")
        indexer.prepare_extraction_records("a-settings-probe-0123456789ab")
        before = sorted(engine.get_aliases_by_index("*extraction*"))
        indexer.check_index_creation("a")
        assert sorted(engine.get_aliases_by_index("*extraction*")) == before

    def test_probes_a_stopped_run_left_are_deleted_first(self, tmp_path) -> None:
        # A run stopped between the probe's create and delete (SIGTERM: the
        # stop fence refuses the delete), or whose delete failed, left the
        # probe for good: nothing else deletes it.
        engine, _ = _strict_engine(tmp_path / "vector.sqlite3", lambda _: None)
        indexer = _probing_indexer(Config(), engine)
        delete = engine.delete_indices

        def lost(names):
            raise RuntimeError("connection reset")

        engine.delete_indices = lost  # type: ignore[method-assign]
        indexer.check_index_creation("x")  # the run goes on
        assert len(engine.get_aliases_by_index("*settings-probe*")) == (
            len(indexer._specs()) + 1
        )

        engine.delete_indices = delete  # type: ignore[method-assign]
        _probing_indexer(Config(), engine).check_index_creation("x")
        assert not engine.get_aliases_by_index("*settings-probe*")

    def test_a_probe_created_before_a_failure_is_deleted(self, tmp_path) -> None:
        # The create's response lost after the index was made (or a stop
        # raised in it): the probe is deleted all the same.
        engine, _ = _strict_engine(tmp_path / "vector.sqlite3", lambda _: None)
        create = engine.create_index

        def lost_response(index, body, *, exist_ok=True):
            create(index, body, exist_ok=exist_ok)
            raise TimeoutError("no answer")

        engine.create_index = lost_response  # type: ignore[method-assign]
        with pytest.raises(TimeoutError):
            _probing_indexer(Config(), engine).check_index_creation("x")
        assert not engine.get_aliases_by_index("*settings-probe*")

    @pytest.mark.parametrize(
        ("error", "blamed", "named"),
        [
            ("unknown setting [index.refresh_intervl]", True, "index_settings"),
            (
                "validation_exception: Validation Failed: 1: this action would "
                "add [1] total shards, but this cluster currently has "
                "[1000]/[1000] maximum shards open;",
                False,
                "max_shards_per_node",
            ),
        ],
    )
    def test_the_error_blames_the_settings_only_when_they_are_at_fault(
        self, tmp_path, error: str, blamed: bool, named: str
    ) -> None:
        # A shard limit was reported as a refused index_settings key.
        from unified_kg_rag.shared.exceptions import StoreRequestRejectedError

        engine, _ = _strict_engine(tmp_path / "vector.sqlite3", lambda _: error)
        for check in (
            lambda ix: ix.check_index_creation("x"),
            lambda ix: ix.prepare_extraction_records("x"),
        ):
            with pytest.raises(StoreRequestRejectedError) as raised:
                check(_probing_indexer(Config(), engine))
            assert named in str(raised.value)
            assert ("index_settings" in str(raised.value)) is blamed

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

    def test_retain_keeps_only_the_named_records_of_the_suffix(self, tmp_path) -> None:
        indexer = _indexer(tmp_path / "vector.sqlite3")
        indexer.write_extraction_records(
            {"k1": {"a": 1}, "k2": {"a": 2}, "k3": {"a": 3}}, suffix="x"
        )
        indexer.write_extraction_records({"k2": {"a": 2}}, suffix="y")

        kept = indexer.retain_extraction_records(["k1", "k3", "gone"], suffix="x")

        assert kept.successful_items == 2
        assert indexer.read_extraction_records(["k1", "k2", "k3"], suffix="x") == {
            "k1": {"a": 1},
            "k3": {"a": 3},
        }
        assert indexer.read_extraction_records(["k2"], suffix="y") == {"k2": {"a": 2}}
        # The old index went with the dropped records; one live index is left.
        alias = indexer._get_name(
            indexer.opensearch_config.extraction_records_index_prefix, "x"
        )
        assert len(indexer.opensearch_client.get_aliases_by_index(f"{alias}-*")) == 1
        # Records written afterwards land in the live index.
        indexer.write_extraction_records({"k4": {}}, suffix="x")
        assert indexer.read_extraction_records(["k4"], suffix="x") == {"k4": {}}

    def test_a_failed_retain_keeps_every_record(self, tmp_path, mocker) -> None:
        indexer = _indexer(tmp_path / "vector.sqlite3")
        indexer.write_extraction_records({"k1": {}, "k2": {}}, suffix="x")
        mocker.patch.object(
            indexer.opensearch_client, "bulk_index", side_effect=RuntimeError("full")
        )
        with pytest.raises(RuntimeError):
            indexer.retain_extraction_records(["k1"], suffix="x")
        assert set(indexer.read_extraction_records(["k1", "k2"], suffix="x")) == {
            "k1",
            "k2",
        }
        alias = indexer._get_name(
            indexer.opensearch_config.extraction_records_index_prefix, "x"
        )
        assert len(indexer.opensearch_client.get_aliases_by_index(f"{alias}-*")) == 1

    def test_a_failed_write_is_reported(self, tmp_path, mocker) -> None:
        indexer = _indexer(tmp_path / "vector.sqlite3")
        mocker.patch.object(
            indexer.opensearch_client, "bulk_index", side_effect=RuntimeError("full")
        )
        stats = indexer.write_extraction_records({"k1": {}}, suffix="x")
        assert isinstance(stats, IndexingStats)
        assert (stats.failed_items, stats.successful_items) == (1, 0)


def _pipeline(config: Config, indexer) -> Any:
    """A pipeline shell whose providers build ``indexer``."""
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline

    pipe = object.__new__(DataIngestionPipeline)
    pipe.config = config
    pipe.providers = SimpleNamespace(build_vector_indexer=lambda: indexer)
    return pipe


def _run_context(incremental: bool = False) -> Any:
    return SimpleNamespace(
        incremental_delta=object() if incremental else None,
        documents=[object()],
        stage_results=[],
        pipeline_id="pid",
    )


class TestRunIndexCreationCheck:
    """``DataIngestionPipeline._check_index_creation``: once per run, before
    the first stage that may call a model, on the run's suffix."""

    @pytest.mark.parametrize("incremental", [False, True])
    def test_the_check_runs_on_the_runs_suffix_and_closes_its_indexer(
        self, mocker, incremental: bool
    ) -> None:
        config = Config()
        config.processing.document_parsing.index_value = "tenant-a"
        indexer = mocker.MagicMock()
        _pipeline(config, indexer)._check_index_creation(_run_context(incremental))

        indexer.check_index_creation.assert_called_once_with(
            "tenant-a", rebuild=not incremental
        )
        if incremental:
            indexer.check_index_settings.assert_called_once_with("tenant-a")
        else:
            indexer.check_index_settings.assert_not_called()
        indexer.close.assert_called_once_with()

    def test_a_failed_check_is_logged_and_its_indexer_closed(
        self, mocker, caplog
    ) -> None:
        indexer = mocker.MagicMock()
        indexer.check_index_creation.side_effect = RuntimeError("connection reset")
        with caplog.at_level("WARNING"):
            _pipeline(Config(), indexer)._check_index_creation(_run_context())
        assert "connection reset" in caplog.text
        indexer.close.assert_called_once_with()

    @pytest.mark.parametrize("stale", [False, True])
    def test_a_lost_lease_while_deleting_probes_stops_the_run(
        self, tmp_path, stale: bool
    ) -> None:
        # The sweep of a stopped run's probes (stale) or the delete of this
        # run's own: a run that lost its lease must stop, not log and go on.
        from unified_kg_rag.shared import NamespaceLeaseError

        engine, _ = _strict_engine(tmp_path / "vector.sqlite3", lambda _: None)
        delete = engine.delete_indices
        if stale:
            engine.delete_indices = lambda names: None  # type: ignore[method-assign]
            _probing_indexer(Config(), engine).check_index_creation()
            assert engine.get_aliases_by_index("*settings-probe*")
        deletes: list[list[str]] = []

        def lost(names):
            deletes.append(list(names))
            raise NamespaceLeaseError("lease lost")

        engine.delete_indices = lost  # type: ignore[method-assign]
        pipe = _pipeline(Config(), _probing_indexer(Config(), engine))
        with pytest.raises(NamespaceLeaseError):
            pipe._check_index_creation(_run_context())
        # Raised by the first delete: the sweep's, or the run's own probes'.
        assert len(deletes) == 1
        engine.delete_indices = delete  # type: ignore[method-assign]

    def test_a_failed_sweep_lets_the_run_check_its_own_probes(self, tmp_path) -> None:
        engine, created = _strict_engine(tmp_path / "vector.sqlite3", lambda _: None)

        def unreachable(pattern):
            raise RuntimeError("connection reset")

        engine.get_aliases_by_index = unreachable  # type: ignore[method-assign]
        indexer = _probing_indexer(Config(), engine)
        indexer.check_index_creation("x")
        assert len(created) == len(indexer._specs()) + 1

    def test_a_refusal_keeps_the_engines_reason(self, tmp_path) -> None:
        from unified_kg_rag.shared.exceptions import StoreRequestRejectedError

        engine, _ = _strict_engine(
            tmp_path / "vector.sqlite3", lambda _: "unknown setting [index.x]"
        )
        with pytest.raises(StoreRequestRejectedError) as raised:
            _probing_indexer(Config(), engine).check_index_creation("x")
        assert raised.value.reason == "illegal_argument_exception"

    def test_the_check_runs_once_before_the_first_model_stage(self, mocker) -> None:
        from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline

        pipe = object.__new__(DataIngestionPipeline)
        pipe._stop_signal = None
        pipe._checkpointing = False
        pipe.s3_cache_manager = None
        pipe.llm_response_cache = None
        names = [
            "document_parsing",
            "document_loading",
            "chunking",
            "graph_extraction",
            "indexing",
        ]
        pipe.stages = [SimpleNamespace(name=name) for name in names]
        order: list[str] = []
        mocker.patch.object(
            pipe,
            "_execute_stage",
            side_effect=lambda stage, context: order.append(stage.name)
            or SimpleNamespace(status=None),
        )
        mocker.patch.object(
            pipe,
            "_check_index_creation",
            side_effect=lambda context: order.append("check"),
        )
        mocker.patch.object(pipe, "_emit_stage_metrics")
        mocker.patch.object(pipe, "_should_stop_pipeline", return_value=False)
        mocker.patch.object(pipe, "_start_index", return_value=0)
        pipe._execute_pipeline_stages(_run_context(), None)
        assert order == [
            "document_parsing",
            "document_loading",
            "check",
            "chunking",
            "graph_extraction",
            "indexing",
        ]
