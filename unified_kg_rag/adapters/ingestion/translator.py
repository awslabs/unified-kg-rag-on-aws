# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import time
from collections import defaultdict
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from unified_kg_rag.adapters.ingestion.chunker import BoundaryRecords
from unified_kg_rag.adapters.llm.chain_factory import setup_chain, usage_model_id
from unified_kg_rag.adapters.llm.langchain import BATCH_ITEM_FAILED, BatchProcessor
from unified_kg_rag.adapters.llm.retry import (
    item_error_predicate,
    item_failed_indices,
    transient_error_predicate,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.ingestion.cache_keys import translation_fingerprint
from unified_kg_rag.domain.ingestion.contributions import (
    RECORD_KEYS_ATTRIBUTE,
    TRANSLATION_RECORD_KIND,
    record_key,
    record_keys,
)
from unified_kg_rag.domain.models import (
    Config,
    LanguageCode,
    ModelPurpose,
    ModelRole,
    TextUnit,
    language_name,
    model_id_value,
)
from unified_kg_rag.domain.prompts import TextTranslationPrompt
from unified_kg_rag.ports.indexer import BaseIndexer
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.logging import log_failure
from unified_kg_rag.shared.usage import record_cache_hit

logger = get_logger(__name__)


class TranslationStats(BaseModel):
    num_total_units: int = Field(
        default=0, description="Total number of text units processed for translation"
    )
    num_successful_translations: int = Field(
        default=0, description="Number of text units that were successfully translated"
    )
    num_failed_translations: int = Field(
        default=0,
        description="Number of text units that encountered translation errors",
    )
    num_translated: int = Field(
        default=0,
        description="Number of text units that actually underwent translation",
    )
    num_reused: int = Field(
        default=0,
        description="Number of text units whose stored translations were "
        "reused (no model call)",
    )
    total_processing_time: float = Field(
        default=0.0, description="Total time spent processing translations (in seconds)"
    )
    failed_text_unit_ids: list[str] = Field(
        default_factory=list,
        description="Ids of text units missing a translation in some target "
        "language, each listed once. They keep their original text, so an "
        "incremental run records their documents FAILED and retries them.",
    )
    item_failed_text_unit_ids: list[str] = Field(
        default_factory=list,
        description="Those of them that failed for an error of their own (an "
        "empty answer, a request rejected for its content), counted toward "
        "indexing.max_document_failures",
    )

    @property
    def processed_unit_count(self) -> int:
        return self.num_successful_translations + self.num_failed_translations

    @property
    def average_processing_time(self) -> float:
        if self.processed_unit_count == 0:
            return 0.0
        return self.total_processing_time / self.processed_unit_count

    @property
    def success_rate(self) -> float:
        if self.num_total_units == 0:
            return 0.0
        return (self.num_successful_translations / self.num_total_units) * 100


class TextUnitTranslator:
    def __init__(
        self,
        config: Config,
        show_progress: bool = True,
        *,
        providers: Providers | None = None,
        records: BoundaryRecords | None = None,
    ) -> None:
        """``records`` (the extraction store) keeps each unit's translations
        by its text and the translation settings
        (``cache_keys.translation_fingerprint``): a unit with stored ones
        takes them instead of a model call, and the others' are stored."""
        self.config = config
        self.records = records
        self._fingerprint: str | None = None
        self.translation_config = config.processing.translation
        self.target_language = self.translation_config.target_language
        self.source_language = self.translation_config.source_language
        # Target first, each language once (a repeated one would be
        # translated, and counted, twice).
        self.all_target_languages = list(
            dict.fromkeys(
                [
                    self.target_language,
                    *(self.translation_config.additional_target_languages or []),
                ]
            )
        )
        self.providers = Providers.resolve(config, providers)
        self.ignore_errors = config.processing.ignore_errors
        self.show_progress = show_progress
        self.stats: TranslationStats | None = None

        self.factory = self.providers.llm_factory
        self.batch_processor = BatchProcessor.from_config(
            self.config,
            is_transient_error=transient_error_predicate(self.factory),
        )
        self.is_item_error = item_error_predicate(self.factory)

        self.translator = setup_chain(
            model_purpose=ModelPurpose.INGESTION,
            factory=self.factory,
            model_id=self.config.model_id(ModelRole.TRANSLATION),
            prompt_class=TextTranslationPrompt,
            min_output_tokens=TextTranslationPrompt.output_floor(config),
            parser=StrOutputParser(),
            custom_prompts=config.custom_prompts,
        )

    def translate_text_units(self, text_units: list[TextUnit]) -> list[TextUnit]:
        if not text_units:
            logger.info("No text units to translate")
            return text_units

        start_time = time.time()
        total_translation_tasks = len(text_units) * len(self.all_target_languages)
        self.stats = TranslationStats(num_total_units=total_translation_tasks)

        logger.info(
            "Starting translation of %s text units to %s language(s): %s",
            len(text_units),
            len(self.all_target_languages),
            ", ".join([lang.value for lang in self.all_target_languages]),
        )

        reused = self._reuse_stored(text_units)
        pending = [unit for unit in text_units if unit.id not in reused]
        if pending:
            for target_language in self.all_target_languages:
                logger.info("Translating to '%s'...", target_language.value)
                self._translate_text_units_batch(pending, target_language)
            self._store(pending)

        self.stats.total_processing_time = time.time() - start_time
        self._log_completion_summary(self.stats)
        return text_units

    @property
    def _model_languages(self) -> list[LanguageCode]:
        """The languages a model call translates into (not the source's)."""
        return [
            lang for lang in self.all_target_languages if lang != self.source_language
        ]

    def record_key(self, unit: TextUnit) -> str:
        """The record key of ``unit``'s translations: its text under the
        translation settings (``translation_fingerprint``)."""
        if self._fingerprint is None:
            self._fingerprint = translation_fingerprint(self.config)
        return record_key(self._fingerprint, unit.text)

    @staticmethod
    def _name_record(unit: TextUnit, key: str) -> None:
        """Name the record of ``unit``'s translations on it, so the store
        keeps the record while the unit is indexed."""
        unit.attributes = {
            **(unit.attributes or {}),
            RECORD_KEYS_ATTRIBUTE: {**record_keys(unit), TRANSLATION_RECORD_KIND: key},
        }

    def _reuse_stored(self, text_units: list[TextUnit]) -> set[str]:
        """Give the units with stored translations in every language they
        are translated into those translations; returns their ids."""
        languages = self._model_languages
        if self.records is None or not languages:
            return set()
        keys = {
            unit.id: (BaseIndexer.get_suffix(unit), self.record_key(unit))
            for unit in text_units
        }
        wanted: dict[str, set[str]] = defaultdict(set)
        for suffix, key in keys.values():
            wanted[suffix].add(key)
        stored = self.records.read_records(
            {suffix: sorted(found) for suffix, found in wanted.items()}
        )
        reused: set[str] = set()
        for unit in text_units:
            suffix, key = keys[unit.id]
            record = stored.get(suffix, {}).get(key)
            texts = record.get("translations") if isinstance(record, dict) else None
            if not isinstance(texts, dict) or not all(
                isinstance(texts.get(lang.value), str) and texts[lang.value].strip()
                for lang in languages
            ):
                continue
            unit.translated_texts = {
                **(unit.translated_texts or {}),
                **{
                    lang.value: (
                        unit.text if lang == self.source_language else texts[lang.value]
                    )
                    for lang in self.all_target_languages
                },
            }
            self._name_record(unit, key)
            reused.add(unit.id)
        if reused:
            logger.info(
                "Reused the stored translations of %d of %d text units",
                len(reused),
                len(text_units),
            )
            if self.stats:
                self.stats.num_reused += len(reused)
                self.stats.num_successful_translations += len(reused) * len(
                    self.all_target_languages
                )
            # Model calls the run did not make (shared.usage), under the id
            # the calls are recorded by (the resolved profile).
            model = usage_model_id(
                self.factory,
                model_id_value(self.config.model_id(ModelRole.TRANSLATION)),
            )
            for _ in range(len(reused) * len(languages)):
                record_cache_hit(ModelPurpose.INGESTION.value, model)
        return reused

    def _store(self, text_units: list[TextUnit]) -> None:
        """Store the translations of the units translated into every
        language, and name each record on its unit."""
        languages = self._model_languages
        if self.records is None or not languages:
            return
        failed = set(self.stats.failed_text_unit_ids) if self.stats else set()
        new_records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for unit in text_units:
            texts = unit.translated_texts or {}
            if unit.id in failed or not all(lang.value in texts for lang in languages):
                continue
            key = self.record_key(unit)
            new_records[BaseIndexer.get_suffix(unit)][key] = {
                "translations": {lang.value: texts[lang.value] for lang in languages}
            }
            self._name_record(unit, key)
        if new_records:
            self.records.write_records(new_records)

    def _translate_text_units_batch(
        self, text_units: list[TextUnit], target_language: LanguageCode
    ) -> None:
        if target_language == self.source_language:
            # Already in that language: the text is its own translation, kept
            # under the language like any other so readers need no special case.
            for text_unit in text_units:
                text_unit.translated_texts = {
                    **(text_unit.translated_texts or {}),
                    target_language.value: text_unit.text,
                }
            if self.stats:
                self.stats.num_successful_translations += len(text_units)
            return

        def prepare_inputs(units: list[TextUnit]) -> list[dict[str, Any]]:
            return self._create_chain_inputs([u.text for u in units], target_language)

        errors: dict[int, BaseException] = {}
        try:
            translation_results = self.batch_processor.execute_with_fallback(
                items_to_process=text_units,
                prepare_inputs_func=prepare_inputs,
                sequential_func=self._translate,
                task_name=f"Translation ({target_language.value})",
                show_progress=self.show_progress,
                errors=errors,
            )
        except Exception as e:
            if not self.ignore_errors:
                raise
            log_failure(
                logger,
                logging.ERROR,
                "Translation to '%s' failed: %s",
                target_language.value,
                error=e,
                traceback=True,
            )
            # Count the whole batch as failed so a fully-failed language does not
            # read as a quiet success.
            for text_unit in text_units:
                self._record_failure(text_unit)
            return

        for text_unit, result in zip(text_units, translation_results, strict=True):
            self._apply_translation_result(
                text_unit,
                None if result is BATCH_ITEM_FAILED else result,
                target_language,
            )
        own = item_failed_indices(
            errors,
            text_units,
            self.is_item_error,
            f"Translation ({target_language.value})",
            self.batch_processor,
            prepare_inputs,
            self._translate,
        )
        if self.stats:
            self.stats.item_failed_text_unit_ids = list(
                dict.fromkeys(
                    [
                        *self.stats.item_failed_text_unit_ids,
                        *(text_units[idx].id for idx in own),
                    ]
                )
            )

    def _translate(self, request: dict[str, Any]) -> str:
        """The chain's translation of ``request``; an empty answer raises
        ``OutputParserException``, as output that does not parse does
        (retried, and an error of the unit's own)."""
        result = str(self.translator.invoke(request))
        if not result.strip():
            raise OutputParserException("The model returned an empty translation")
        return result

    @staticmethod
    def _create_chain_inputs(
        texts: list[str], target_language: LanguageCode
    ) -> list[dict[str, Any]]:
        # The prompt names the language ("Korean"): the enum formats as
        # "LanguageCode.KO" in a template.
        name = language_name(target_language)
        return [{"text": text, "target_language": name} for text in texts]

    def _apply_translation_result(
        self, text_unit: TextUnit, result: str | None, target_language: LanguageCode
    ) -> None:
        # A blank answer already failed its call (``_translate``).
        if result is None:
            self._record_failure(text_unit)
            return

        if text_unit.translated_texts is None:
            text_unit.translated_texts = {}
        text_unit.translated_texts[target_language.value] = result.strip()

        if self.stats:
            self.stats.num_translated += 1
            self.stats.num_successful_translations += 1

    def _record_failure(self, text_unit: TextUnit) -> None:
        if self.stats is None:
            return
        self.stats.num_failed_translations += 1
        if text_unit.id not in self.stats.failed_text_unit_ids:
            self.stats.failed_text_unit_ids.append(text_unit.id)

    @staticmethod
    def _log_completion_summary(stats: TranslationStats) -> None:
        if not stats:
            return

        logger.info(
            "Translation completed - Total time: %.2fs, "
            "Success rate: %.2f%% "
            "(%s/%s)",
            stats.total_processing_time,
            stats.success_rate,
            stats.num_successful_translations,
            stats.num_total_units,
        )

        if stats.num_failed_translations > 0:
            logger.warning(
                "Translation issues: %s units failed", stats.num_failed_translations
            )
