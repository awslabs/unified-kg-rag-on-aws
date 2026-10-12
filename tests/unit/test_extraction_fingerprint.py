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
    legacy_doc_id,
    registry_scope,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    DocStatusRecord,
    Document,
    LanguageCode,
    PipelineStageType,
)
from unified_kg_rag.shared.utils.cache_keys import (
    extraction_fingerprint,
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
        _set("processing.graph_extraction.extraction_model_id", "amazon.nova-pro"),
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
        # No identifier: nothing is guarded whatever the scope.
        {"apply_to": "all"},
    ],
)
def test_a_guardrail_that_skips_ingestion_does_not_move_it(
    fields: dict[str, object],
) -> None:
    assert extraction_fingerprint(_guardrail(**fields)) == extraction_fingerprint(
        Config()
    )


def test_a_guardrail_that_guards_ingestion_moves_it() -> None:
    guarded = extraction_fingerprint(_guardrail(identifier="gr-1", apply_to="all"))
    assert guarded != extraction_fingerprint(Config())
    assert guarded != extraction_fingerprint(
        _guardrail(identifier="gr-2", apply_to="all")
    )
    assert guarded != extraction_fingerprint(
        _guardrail(identifier="gr-1", version="2", apply_to="all")
    )
    # Trace details do not change what the guarded model returns.
    assert guarded == extraction_fingerprint(
        _guardrail(identifier="gr-1", apply_to="all", trace=True)
    )


@pytest.mark.parametrize(
    "block, field, value",
    [
        ("processing.translation", "translation_model_id", "amazon.nova-pro"),
        ("processing.gleaning", "max_rounds", 1),
        ("fixing", "fixing_model_id", "amazon.nova-pro"),
        (
            "processing.graph_extraction.description_summarization",
            "max_summary_tokens",
            64,
        ),
    ],
)
def test_settings_of_a_disabled_block_do_not_move_it(
    block: str, field: str, value: object
) -> None:
    disabled = Config()
    _set(f"{block}.enabled", False)(disabled)
    before = extraction_fingerprint(disabled)
    _set(f"{block}.{field}", value)(disabled)
    assert extraction_fingerprint(disabled) == before

    # Switched on, the same field shapes the output again.
    enabled = Config()
    before = extraction_fingerprint(enabled)
    _set(f"{block}.{field}", value)(enabled)
    assert extraction_fingerprint(enabled) != before


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


# The default configuration's fingerprint, as recorded on doc-status records
# by released versions. It must not move, or upgrading re-extracts every
# incrementally indexed corpus.
_RELEASED_DEFAULT_FINGERPRINT = "0b3882fff6d2"


def test_default_fingerprint_is_unchanged() -> None:
    assert extraction_fingerprint(Config()) == _RELEASED_DEFAULT_FINGERPRINT


@pytest.mark.parametrize(
    "change",
    [
        _set("processing.claim_extraction.max_entities_per_prompt", 5),
        _set("processing.claim_extraction.extraction_model_id", "amazon.nova-pro"),
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
        file_type="txt",
        total_pages=1,
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


class _ContentOnlyStore(FakeDocStatusStore):
    """A registry written against the port before diff took a fingerprint."""

    def diff(  # type: ignore[override]
        self, incoming: dict[str, str], scope: str | None = None
    ):
        return super().diff(incoming, scope=scope)


def test_store_without_fingerprint_support_is_diffed_by_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _ContentOnlyStore()
    document = _doc("/a.txt", "a")
    store.put(_record(document, extraction_fingerprint="old"))

    with caplog.at_level("WARNING"):
        delta, _ = detect_delta([document], store, extraction_fingerprint="new")

    assert delta.unchanged == [document_doc_id(document)]
    assert "takes no extraction_fingerprint" in caplog.text


def test_adopted_legacy_record_of_other_settings_reads_changed() -> None:
    store = FakeDocStatusStore()
    document = _doc("/a.txt", "a")
    document.metadata["registry_source"] = "/corpus"
    scope = registry_scope("default", "/corpus")
    # Keyed without the source scope, as before scopes joined the key.
    store.put(
        _record(document, extraction_fingerprint="old").model_copy(
            update={"doc_id": legacy_doc_id(document), "scope": scope}
        )
    )

    delta, _ = detect_delta(
        [document],
        store,
        scope=scope,
        legacy_doc_ids={document_doc_id(document): legacy_doc_id(document)},
        extraction_fingerprint="new",
    )

    assert delta.changed == delta.settings_changed == [document_doc_id(document)]
    assert store.get(document_doc_id(document)) is not None
