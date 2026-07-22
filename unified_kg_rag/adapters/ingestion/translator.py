# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from typing import Any

import boto3
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from unified_kg_rag.adapters.aws import BedrockLanguageModelFactory
from unified_kg_rag.adapters.aws.chain_factory import setup_chain
from unified_kg_rag.domain.models import Config, LanguageCode, TextUnit
from unified_kg_rag.domain.prompts import TextTranslationPrompt
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import BatchProcessor

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
        self.boto_session = boto_session or boto3.Session(
            profile_name=config.aws.profile_name
        )
        self.ignore_errors = config.processing.ignore_errors
        self.show_progress = show_progress
        self.stats: TranslationStats | None = None

        self.factory = BedrockLanguageModelFactory(
            config=config,
            boto_session=self.boto_session,
            region_name=config.aws.bedrock.region_name,
        )
        self.batch_processor = BatchProcessor()

        self.translator = setup_chain(
            factory=self.factory,
            model_id=self.translation_config.translation_model_id,
            prompt_class=TextTranslationPrompt,
            parser=StrOutputParser(),
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

        try:
            for target_language in self.all_target_languages:
                logger.info("Translating to '%s'...", target_language.value)
                self._translate_text_units_batch(text_units, target_language)

            self.stats.total_processing_time = time.time() - start_time
            self._log_completion_summary(self.stats)
        except Exception as e:
            logger.exception("Translation failed: %s", e)

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
                batch_func=self.translator.batch,
                sequential_func=self.translator.invoke,
                task_name=f"Translation ({target_language.value})",
                run_config=self.config.processing.model_dump(),
                show_progress=self.show_progress,
            )
        except Exception as e:
            logger.error(
                "Translation to '%s' failed: %s",
                target_language.value,
                e,
                exc_info=True,
            )
            # Count the whole batch as failed so a fully-failed language does not
            # read as a quiet success: num_total_units was preset, so without
            # this the success rate drops but num_failed_translations stays 0 and
            # the "X units failed" summary warning never fires.
            if self.stats is not None:
                self.stats.num_failed_translations += len(text_units)
            return

        for text_unit, result in zip(text_units, translation_results, strict=True):
            self._apply_translation_result(text_unit, result, target_language)

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
            if self.stats:
                self.stats.num_failed_translations += 1
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
            if self.stats:
                self.stats.num_failed_translations += 1

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
