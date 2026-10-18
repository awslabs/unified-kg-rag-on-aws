# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The extraction fingerprint recorded per document, and the delta it drives."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.domain.ingestion.delta_detector import (
    compute_content_hash,
    detect_delta,
    document_doc_id,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    DocStatusRecord,
    Document,
    ExtractionSettings,
    LanguageCode,
    PipelineStageType,
)
from unified_kg_rag.shared.utils.cache_keys import (
    changed_settings,
    extraction_fingerprint,
    extraction_settings,
    stage_input_fingerprint,
)

pytestmark = pytest.mark.unit


def _set(path: str, value: object) -> Callable[[Config], None]:
    def apply(config: Config) -> None:
        *parents, leaf = path.split(".")
        node: object = config
        for name in parents:
            node = getattr(node, name)
        setattr(node, leaf, value)

    return apply


@pytest.mark.parametrize(
    "change",
    [
        _set("custom_prompts.graph_extraction_system", "tuned {input_text}"),
        _set("processing.graph_extraction.entity_types", ["PRODUCT: goods"]),
        _set("models.roles.graph_extraction", "amazon.nova-pro"),
        _set("processing.chunking.max_chunk_size", 4000),
        _set("processing.gleaning.max_rounds", 1),
        _set("processing.similarity_threshold", 0.5),
        _set("processing.translation.enabled", False),
    ],
)
def test_changes_with_the_extraction_settings(
    change: Callable[[Config], None],
) -> None:
    config = Config()
    before = extraction_fingerprint(config)
    change(config)
    assert extraction_fingerprint(config) != before


@pytest.mark.parametrize(
    "change",
    [
        # Where the corpus lives and how it is named is the document's
        # identity, not its extraction.
        _set("processing.document_parsing.source_directory", "/elsewhere"),
        _set("processing.document_parsing.source_scope", "s3://bucket/corpus"),
        _set("processing.document_parsing.index_value", "tenant"),
        _set("processing.ignore_errors", True),
        _set("processing.deduplicate", True),
        # A reset only loads the whole corpus instead of the delta.
        _set("indexing.reset", True),
        # Later stages do not shape a document's entities.
        _set("custom_prompts.community_report_system", "tuned"),
        _set("custom_prompts.claim_extraction_system", "tuned"),
        _set("indexing.max_failure_rate", 0.5),
    ],
)
def test_ignores_settings_that_do_not_shape_a_documents_artifacts(
    change: Callable[[Config], None],
) -> None:
    config = Config()
    before = extraction_fingerprint(config)
    change(config)
    assert extraction_fingerprint(config) == before


def _guardrail(**fields: object) -> Config:
    return Config.model_validate({"aws": {"bedrock": {"guardrail": fields}}})


@pytest.mark.parametrize(
    "fields",
    [
        {"identifier": "gr-query"},
        {"identifier": "gr-query", "version": "3"},
        {"identifier": "gr-query", "trace": True},
    ],
)
def test_the_query_guardrail_does_not_move_it(
    fields: dict[str, object],
) -> None:
    assert extraction_fingerprint(_guardrail(**fields)) == extraction_fingerprint(
        Config()
    )


@pytest.mark.parametrize(
    "block, path, value",
    [
        ("processing.translation", "models.roles.translation", "amazon.nova-pro"),
        ("processing.gleaning", "processing.gleaning.max_rounds", 1),
        ("processing.gleaning", "models.roles.gleaning", "amazon.nova-pro"),
        ("fixing", "models.roles.fixing", "amazon.nova-pro"),
        (
            "processing.graph_extraction.description_summarization",
            "processing.graph_extraction.description_summarization.max_summary_tokens",
            64,
        ),
    ],
)
def test_settings_of_a_disabled_block_do_not_move_it(
    block: str, path: str, value: object
) -> None:
    # A role's model lives in models.roles, outside its stage's block, and
    # still counts only while that stage runs.
    disabled = Config()
    _set(f"{block}.enabled", False)(disabled)
    before = extraction_fingerprint(disabled)
    _set(path, value)(disabled)
    assert extraction_fingerprint(disabled) == before

    # Switched on, the same field shapes the output again.
    enabled = Config()
    before = extraction_fingerprint(enabled)
    _set(path, value)(enabled)
    assert extraction_fingerprint(enabled) != before


def test_a_tier_change_moves_it_through_the_roles_on_that_tier() -> None:
    # Graph extraction runs on the default tier: moving the tier moves the
    # extraction settings, though no role key changed.
    config = Config()
    before = extraction_fingerprint(config)
    config.models.default = "amazon.nova-pro"
    assert extraction_fingerprint(config) != before
    # Pinning the role back to the old model restores the fingerprint.
    config.models.roles.graph_extraction = Config().tier_model_id("default")
    config.models.roles.gleaning = Config().tier_model_id("default")
    config.models.roles.fixing = Config().tier_model_id("default")
    assert extraction_fingerprint(config) == before


@pytest.mark.parametrize(
    "field, value",
    [
        # Extraction, gleaning, claim and description prompts write in it.
        ("target_language", LanguageCode.KO),
        # The indexer picks the chunk analyzer and translated-text fields
        # from the languages.
        ("source_language", LanguageCode.KO),
        ("additional_target_languages", [LanguageCode.JA]),
    ],
)
def test_languages_count_while_translation_is_off(field: str, value: object) -> None:
    config = Config()
    config.processing.translation.enabled = False
    before = extraction_fingerprint(config)
    indexing_before = stage_input_fingerprint(config, PipelineStageType.INDEXING)
    _set(f"processing.translation.{field}", value)(config)
    assert extraction_fingerprint(config) != before
    assert (
        stage_input_fingerprint(config, PipelineStageType.INDEXING) != indexing_before
    )


def test_grounding_settings_count_only_while_grounding_is_on() -> None:
    config = Config()
    before = extraction_fingerprint(config)
    _set("processing.graph_extraction.entity_grounding.penalty_factor", 0.1)(config)
    assert extraction_fingerprint(config) == before
    _set("processing.graph_extraction.entity_grounding.enabled", True)(config)
    assert extraction_fingerprint(config) != before


@pytest.mark.parametrize(
    "change",
    [
        _set("processing.claim_extraction.max_entities_per_prompt", 5),
        _set("models.roles.claim_extraction", "amazon.nova-pro"),
        _set("custom_prompts.claim_extraction_system", "tuned"),
        _set("custom_prompts.claim_extraction_human", "tuned {input_text}"),
    ],
)
def test_claim_settings_count_only_while_claim_extraction_is_on(
    change: Callable[[Config], None],
) -> None:
    # Claims are a document's own output: switching extraction on, or changing
    # what it produces, re-extracts documents whose content did not change.
    disabled = Config()
    enabled = Config()
    enabled.processing.claim_extraction.enabled = True
    assert extraction_fingerprint(enabled) != extraction_fingerprint(disabled)

    before = extraction_fingerprint(disabled)
    change(disabled)
    assert extraction_fingerprint(disabled) == before

    before = extraction_fingerprint(enabled)
    change(enabled)
    assert extraction_fingerprint(enabled) != before


def test_is_stable_for_an_equal_configuration() -> None:
    assert extraction_fingerprint(Config()) == extraction_fingerprint(Config())


def _doc(path: str, text: str) -> Document:
    return Document(
        page_content=text,
        document_id="x",
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
    )


def _record(document: Document, **fields: object) -> DocStatusRecord:
    return DocStatusRecord.model_validate(
        {
            "doc_id": document_doc_id(document),
            "content_hash": compute_content_hash(document),
            "status": DocStatus.PROCESSED,
            **fields,
        }
    )


def test_delta_reextracts_documents_of_other_settings_and_logs_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = FakeDocStatusStore()
    stale, current, legacy = (
        _doc("/stale.txt", "a"),
        _doc("/current.txt", "b"),
        _doc("/legacy.txt", "c"),
    )
    store.put_many(
        [
            _record(stale, extraction_fingerprint="old"),
            _record(current, extraction_fingerprint="new"),
            _record(legacy),
        ]
    )

    with caplog.at_level("WARNING"):
        delta, _ = detect_delta(
            [stale, current, legacy], store, extraction_fingerprint="new"
        )

    assert delta.changed == delta.settings_changed == [document_doc_id(stale)]
    assert sorted(delta.unchanged) == sorted(
        [document_doc_id(current), document_doc_id(legacy)]
    )
    assert caplog.text.count("Re-extracting 1 documents") == 1


def test_settings_change_retries_a_document_that_used_up_its_retries() -> None:
    store = FakeDocStatusStore()
    exhausted = _doc("/a.txt", "a")
    store.put(
        _record(
            exhausted,
            status=DocStatus.FAILED,
            failure_count=3,
            extraction_fingerprint="old",
        )
    )

    held, _ = detect_delta([exhausted], store, max_failures=3)
    assert held.unchanged == [document_doc_id(exhausted)]

    retried, _ = detect_delta(
        [exhausted], store, max_failures=3, extraction_fingerprint="new"
    )
    assert retried.changed == [document_doc_id(exhausted)]


class TestChangedSettings:
    """Extraction-settings records compare field by field."""

    @staticmethod
    def _settings(config: Config) -> ExtractionSettings:
        return extraction_settings(config)

    def test_names_the_fields_that_changed(self) -> None:
        changed = Config()
        changed.processing.chunking.max_chunk_size += 100
        changed.processing.gleaning.max_rounds = 1
        assert changed_settings(self._settings(Config()), self._settings(changed)) == [
            "processing.chunking.max_chunk_size",
            "processing.gleaning.max_rounds",
        ]
        assert (
            changed_settings(self._settings(Config()), self._settings(Config())) == []
        )

    def test_a_field_the_old_record_lacks_counts_at_its_default(self) -> None:
        # A field a newer version added (its default keeps the old
        # behaviour) does not re-extract a corpus; set away from it, it does.
        new = self._settings(Config())
        path = "processing.graph_extraction.max_entities_per_chunk"
        old = ExtractionSettings(
            version=new.version,
            values={k: v for k, v in new.values.items() if k != path},
        )
        assert old.fingerprint != new.fingerprint
        assert changed_settings(old, new) == []

        tuned = Config()
        tuned.processing.graph_extraction.max_entities_per_chunk += 1
        assert changed_settings(old, self._settings(tuned)) == [path]

    def test_a_removed_field_is_ignored_and_one_out_of_effect_is_not(self) -> None:
        new = self._settings(Config())
        removed = ExtractionSettings(
            version=new.version,
            values={**new.values, "processing.chunking.content_type": "text"},
        )
        assert changed_settings(removed, new) == []

        gleaning = self._settings(Config())
        no_gleaning = self._settings(
            Config.model_validate({"processing": {"gleaning": {"enabled": False}}})
        )
        assert "processing.gleaning.max_rounds" not in no_gleaning.values
        assert changed_settings(gleaning, no_gleaning)
        assert changed_settings(no_gleaning, gleaning)

    def test_another_format_version_differs_as_a_whole(self) -> None:
        new = self._settings(Config())
        old = ExtractionSettings(version=new.version - 1, values=new.values)
        assert changed_settings(old, new) == ["version"]

    def test_long_values_are_recorded_by_digest(self) -> None:
        config = Config()
        config.custom_prompts.graph_extraction_system = "Extract {input_text}. " * 50
        recorded = self._settings(config).values[
            "custom_prompts.graph_extraction_system"
        ]
        assert isinstance(recorded, str) and recorded.startswith("sha256:")
        assert self._settings(config).fingerprint == extraction_fingerprint(config)
