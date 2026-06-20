# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from multiprocessing import cpu_count
from typing import Any

import boto3
from pydantic import BaseModel
from tqdm import tqdm

from aws_graphrag.adapters.aws import BedrockLanguageModelFactory
from aws_graphrag.adapters.aws.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from aws_graphrag.domain.ingestion.base_processor import (
    BaseProcessor,
    check_entity_relevance_task,
)
from aws_graphrag.domain.models import Claim, Config, Entity, TextUnit
from aws_graphrag.domain.prompts import ClaimExtractionPrompt
from aws_graphrag.shared import get_logger
from aws_graphrag.shared.utils import (
    BatchProcessor,
    ensure_list,
    generate_stable_id,
)

logger = get_logger(__name__)


def format_entities_with_limit_task(entities: list[Entity], max_entities: int) -> str:
    if not entities:
        return ""

    if len(entities) <= max_entities:
        return "\n".join([e.name for e in entities])

    sorted_entities = sorted(
        entities,
        key=lambda e: (
            bool(e.description and e.description.strip()),
            len(e.text_unit_ids or []),
        ),
        reverse=True,
    )
    selected_entities = sorted_entities[:max_entities]
    entity_list = "\n".join([e.name for e in selected_entities])

    if len(entities) > max_entities:
        entity_list += f"\n... and {len(entities) - max_entities} more entities"

    return entity_list


def _prepare_claim_input_task(
    unit: TextUnit, all_entities: list[Entity], config: dict[str, Any]
) -> dict[str, Any]:
    if unit.translated_texts:
        target_language = config.get("target_language", "en")
        unit_text = unit.translated_texts.get(target_language, unit.text or "")
    else:
        unit_text = unit.text or ""

    entity_specs_str = ""
    if config["max_entities_per_prompt"] > 0 and all_entities:
        relevant_entities = [
            entity
            for entity in all_entities
            if check_entity_relevance_task(entity, unit.id)[1]
        ]
        entity_specs_str = format_entities_with_limit_task(
            relevant_entities, config["max_entities_per_prompt"]
        )

    return {
        "input_text": unit_text,
        "entity_specs": entity_specs_str,
    }


class ClaimExtractionStats(BaseModel):
    num_total_units: int = 0
    num_successful_extractions: int = 0
    num_failed_extractions: int = 0
    total_claims_extracted: int = 0
    total_processing_time: float = 0.0

    @property
    def processed_unit_count(self) -> int:
        return self.num_successful_extractions + self.num_failed_extractions

    @property
    def success_rate(self) -> float:
        if self.processed_unit_count == 0:
            return 0.0
        return (self.num_successful_extractions / self.processed_unit_count) * 100


class ClaimExtractor(BaseProcessor):
    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        max_workers: int | None = None,
        use_process_pool: bool = True,
        show_progress: bool = True,
    ):
        super().__init__(config)
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name
        )
        self.claim_extraction_config = self.config.processing.claim_extraction
        self.ignore_errors = self.config.processing.ignore_errors
        self.max_workers = max_workers or max(1, int(cpu_count() * 0.8))
        self.use_process_pool = use_process_pool
        self.show_progress = show_progress
        self.batch_processor = BatchProcessor()

        self.factory = BedrockLanguageModelFactory(
            config=config,
            boto_session=self.boto_session,
            region_name=self.config.aws.bedrock.region_name,
        )

        robust_xml_output_parser = create_robust_xml_output_parser(
            factory=self.factory,
            enable_output_fixing=self.config.fixing.enabled,
            output_fixing_model_id=self.config.fixing.fixing_model_id,
        )
        self.claim_extractor = setup_chain(
            factory=self.factory,
            model_id=self.claim_extraction_config.extraction_model_id,
            prompt_class=ClaimExtractionPrompt,
            parser=robust_xml_output_parser,
            custom_prompts=self.config.custom_prompts,
        )

        self.stats: ClaimExtractionStats = ClaimExtractionStats()
        logger.info(
            "ClaimExtractor initialized with %s workers, process pool: %s",
            self.max_workers,
            self.use_process_pool,
        )

    def extract_from_text_units(
        self, text_units: list[TextUnit], entities: list[Entity] | None = None
    ) -> tuple[list[Claim], ClaimExtractionStats]:
        if not text_units:
            logger.warning("No text units provided for claim extraction")
            return [], ClaimExtractionStats()

        all_entities = entities or []
        logger.info(
            "Starting claim extraction from %s text units with %s entities for relevance filtering",
            len(text_units),
            len(all_entities),
        )

        self.stats = ClaimExtractionStats(num_total_units=len(text_units))
        start_time = time.time()

        prepared_inputs = self._prepare_extraction_inputs(text_units, all_entities)
        unit_to_input = {
            unit.id: input_data
            for unit, input_data in zip(text_units, prepared_inputs, strict=True)
        }

        def prepare_inputs_for_chunk(
            chunk_items: list[TextUnit],
        ) -> list[dict[str, Any]]:
            return [
                unit_to_input[unit.id]
                for unit in chunk_items
                if unit.id in unit_to_input
            ]

        try:
            extraction_results = self.batch_processor.execute_with_fallback(
                items_to_process=text_units,
                prepare_inputs_func=prepare_inputs_for_chunk,
                batch_func=self.claim_extractor.batch,
                sequential_func=self.claim_extractor.invoke,
                task_name="Claim Extraction",
                run_config=self.config.processing.model_dump(),
                show_progress=self.show_progress,
            )
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.error("Error during claim extraction: %s", e)
            return [], ClaimExtractionStats()

        all_claims = self._process_extraction_results(text_units, extraction_results)
        initial_claim_count = len(all_claims)
        merged_claims = self._merge_claims(all_claims)

        self.stats.total_claims_extracted = len(merged_claims)
        self.stats.total_processing_time = time.time() - start_time

        logger.info(
            "Claim extraction completed: extracted %s claims, merged to %s unique claims",
            initial_claim_count,
            len(merged_claims),
        )
        self._log_completion_summary(self.stats)

        return merged_claims, self.stats

    def _prepare_extraction_inputs(
        self, text_units: list[TextUnit], all_entities: list[Entity]
    ) -> list[dict[str, Any]]:
        config_for_task = {
            "max_entities_per_prompt": self.claim_extraction_config.max_entities_per_prompt,
            "target_language": self.config.processing.translation.target_language.value,
        }

        task_with_args = partial(
            _prepare_claim_input_task,
            all_entities=all_entities,
            config=config_for_task,
        )

        inputs = []
        executor_class = (
            ProcessPoolExecutor if self.use_process_pool else ThreadPoolExecutor
        )

        with executor_class(max_workers=self.max_workers) as executor:
            future_to_unit = {
                executor.submit(task_with_args, unit): unit for unit in text_units
            }

            for future in tqdm(
                as_completed(future_to_unit),
                total=len(text_units),
                desc="Preparing Claim Inputs",
                disable=not self.show_progress,
            ):
                try:
                    result = future.result()
                    inputs.append(result)
                except Exception as e:
                    logger.error("Error preparing input for text unit: %s", e)

        return inputs

    def _process_extraction_results(
        self, text_units: list[TextUnit], extraction_results: list[Any]
    ) -> list[Claim]:
        all_claims = []

        if len(text_units) != len(extraction_results):
            logger.warning(
                "Mismatch in text units (%s) and extraction results (%s). Some results may have been lost during processing. Proceeding with available results.",
                len(text_units),
                len(extraction_results),
            )

        for text_unit, result in zip(text_units, extraction_results, strict=False):
            if result:
                try:
                    claims = self._parse_extraction_result(result, text_unit)
                    all_claims.extend(claims)
                    self.stats.num_successful_extractions += 1
                except Exception as e:
                    self.stats.num_failed_extractions += 1
                    logger.warning(
                        "Failed to parse extraction result for text unit '%s': %s",
                        text_unit.id,
                        e,
                    )
            else:
                self.stats.num_failed_extractions += 1
        return all_claims

    def _parse_extraction_result(
        self, result: dict[str, Any], text_unit: TextUnit
    ) -> list[Claim]:
        if not isinstance(result, dict) or "claims" not in result:
            return []

        claims_data = ensure_list(result.get("claims"), inner_key="claim")
        claims = []
        for claim_data in claims_data:
            if claim := self._parse_claim_data(claim_data, text_unit):
                claims.append(claim)
        return claims

    @staticmethod
    def _get_string_value(data: Any) -> str:
        if isinstance(data, str):
            return data.strip()
        if isinstance(data, dict):
            if "#text" in data:
                return str(data["#text"]).strip()
            if len(data.values()) == 1:
                return ClaimExtractor._get_string_value(list(data.values())[0])
            return str(data)
        if data is None:
            return ""
        return str(data).strip()

    def _parse_claim_data(
        self, claim_data: dict[str, Any], text_unit: TextUnit
    ) -> Claim | None:
        try:
            subject_name = self._get_string_value(claim_data.get("subject"))
            object_name = self._get_string_value(claim_data.get("object"))
            claim_type = self._get_string_value(claim_data.get("claim_type"))

            if not all([subject_name, object_name, claim_type]):
                return None

            claim_id = self._generate_claim_id(subject_name, object_name, claim_type)
            attributes = self._parse_attributes(claim_data.get("attributes"), text_unit)

            return Claim(
                id=claim_id,
                short_id=claim_id[:8],
                subject_id="",
                subject_name=subject_name,
                object_id="",
                object_name=object_name,
                type=claim_type,
                status=self._get_string_value(claim_data.get("claim_status")),
                start_date=self._get_string_value(claim_data.get("start_date")),
                end_date=self._get_string_value(claim_data.get("end_date")),
                description=self._get_string_value(claim_data.get("description")),
                description_embedding=None,
                text_unit_ids=[text_unit.id],
                source_text=self._get_string_value(claim_data.get("source_text")),
                attributes=attributes,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        except Exception as e:
            logger.warning(
                "Error creating claim object from text unit '%s': %s", text_unit.id, e
            )
            return None

    @staticmethod
    def _generate_claim_id(subject_name: str, object_name: str, claim_type: str) -> str:
        claim_id_content = f"claim:{subject_name}:{object_name}:{claim_type}".lower()
        return generate_stable_id(claim_id_content)

    def _merge_claims(self, claims: list[Claim]) -> list[Claim]:
        def safe_list_merge(current: list[str], new: list[str]) -> list[str]:
            merged_current = current if current is not None else []
            merged_new = new if new is not None else []
            return list(set(merged_current + merged_new))

        field_mergers: dict[str, Callable[[Any, Any], Any]] = {
            "description": self._merge_description,
            "source_text": self._merge_description,
            "text_unit_ids": safe_list_merge,
        }
        return self._merge_items(
            items=claims,
            item_name="Claim",
            field_mergers=field_mergers,
            frequency_fields=["type", "status"],
            log_message_formatter=lambda c: (
                f"Claim '{c.subject_name}' -> '{c.object_name}' "
                f"(type: '{c.type}') merged {{count}} instances"
            ),
        )

    @staticmethod
    def _log_completion_summary(stats: ClaimExtractionStats) -> None:
        if not stats:
            return

        logger.info(
            f"Processing completed in {stats.total_processing_time:.2f}s - "
            f"Success rate: {stats.success_rate:.1f}% "
            f"({stats.num_successful_extractions}/{stats.num_total_units} units processed)"
        )

        if stats.num_failed_extractions > 0:
            logger.warning(
                "%s text units failed during extraction", stats.num_failed_extractions
            )
