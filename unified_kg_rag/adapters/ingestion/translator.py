# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from typing import Any

import boto3
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from unified_kg_rag.adapters.aws.bedrock_retry import is_transient_bedrock_error
from unified_kg_rag.adapters.aws.chain_factory import setup_chain
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import Config, LanguageCode, ModelPurpose, TextUnit
from unified_kg_rag.domain.prompts import TextTranslationPrompt
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils.langchain import BATCH_ITEM_FAILED, BatchProcessor

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
    total_processing_time: float = Field(
        default=0.0, description="Total time spent processing translations (in seconds)"
    )
    failed_text_unit_ids: list[str] = Field(
        default_factory=list,
        description="Ids of text units missing a translation in some target "
        "language, each listed once. They keep their original text, so an "
        "incremental run records their documents FAILED and retries them.",
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
        boto_session: boto3.Session | None = None,
        show_progress: bool = True,
        *,
        providers: Providers | None = None,
    ) -> None:
        self.config = config
        self.translation_config = config.processing.translation
        self.target_language = self.translation_config.target_language
        self.additional_target_languages = (
            self.translation_config.additional_target_languages or []
        )
        self.all_target_languages = [
            self.target_language
        ] + self.additional_target_languages
        self.providers = Providers.resolve(config, providers, boto_session)
        self.boto_session = self.providers.boto_session
        self.ignore_errors = config.processing.ignore_errors
        self.show_progress = show_progress
        self.stats: TranslationStats | None = None

        self.factory = self.providers.llm_factory
        self.batch_processor = BatchProcessor(
            is_transient_error=is_transient_bedrock_error
        )

        self.translator = setup_chain(
            model_purpose=ModelPurpose.INGESTION,
            factory=self.factory,
            model_id=self.translation_config.translation_model_id,
            prompt_class=TextTranslationPrompt,
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

        for target_language in self.all_target_languages:
            logger.info("Translating to '%s'...", target_language.value)
            self._translate_text_units_batch(text_units, target_language)

        self.stats.total_processing_time = time.time() - start_time
        self._log_completion_summary(self.stats)
        return text_units

    def _translate_text_units_batch(
        self, text_units: list[TextUnit], target_language: LanguageCode
    ) -> None:
        texts_to_translate = [unit.text for unit in text_units]

        try:
            translation_results = self.batch_processor.execute_with_fallback(
                items_to_process=texts_to_translate,
                prepare_inputs_func=lambda texts: self._create_chain_inputs(
                    texts, target_language
                ),
                sequential_func=self.translator.invoke,
                task_name=f"Translation ({target_language.value})",
                run_config=self.config.processing.model_dump(),
                show_progress=self.show_progress,
            )
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.error(
                "Translation to '%s' failed: %s",
                target_language.value,
                e,
                exc_info=True,
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

    @staticmethod
    def _create_chain_inputs(
        texts: list[str], target_language: LanguageCode
    ) -> list[dict[str, Any]]:
        try:
            return [
                {"text": text, "target_language": target_language} for text in texts
            ]
        except Exception as e:
            logger.error("Failed to create translation inputs: %s", e)
            return []

    def _apply_translation_result(
        self, text_unit: TextUnit, result: str | None, target_language: LanguageCode
    ) -> None:
        if not result or not result.strip():
            self._record_failure(text_unit)
            return

        try:
            if text_unit.translated_texts is None:
                text_unit.translated_texts = {}

            text_unit.translated_texts[target_language.value] = result.strip()

            if self.stats:
                self.stats.num_translated += 1
                self.stats.num_successful_translations += 1
        except Exception as e:
            logger.error(
                "Failed to apply translation for text unit '%s': %s", text_unit.id, e
            )
            self._record_failure(text_unit)

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
