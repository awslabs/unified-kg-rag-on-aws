# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from unified_kg_rag.adapters.llm.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from unified_kg_rag.adapters.llm.retry import transient_error_predicate
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.ingestion.base_processor import (
    BaseProcessor,
    coerce_llm_text,
    format_entities_with_limit_task,
    group_by_text_unit,
)
from unified_kg_rag.domain.ingestion.claim_resolver import claim_identity_suffix
from unified_kg_rag.domain.ingestion.contributions import merge_claim_contributions
from unified_kg_rag.domain.models import (
    Claim,
    Config,
    Entity,
    ModelPurpose,
    ModelRole,
    TextUnit,
)
from unified_kg_rag.domain.prompts import ClaimExtractionPrompt
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import (
    ensure_list,
    entity_key,
    generate_stable_id,
)
from unified_kg_rag.shared.utils.langchain import BATCH_ITEM_FAILED, BatchProcessor

logger = get_logger(__name__)


def prepare_claim_input(
    unit: TextUnit, unit_entities: list[Entity], config: dict[str, Any]
) -> dict[str, Any]:
    """The claim prompt's input for ``unit`` and the entities citing it."""
    if unit.translated_texts:
        target_language = config.get("target_language", "en")
        unit_text = unit.translated_texts.get(target_language, unit.text or "")
    else:
        unit_text = unit.text or ""

    entity_specs_str = ""
    if config["max_entities_per_prompt"] > 0:
        entity_specs_str = format_entities_with_limit_task(
            unit_entities, config["max_entities_per_prompt"]
        )

    return {
        "input_text": unit_text,
        "entity_specs": entity_specs_str,
    }


class ClaimExtractionStats(BaseModel):
    num_total_units: int = 0
    num_successful_extractions: int = 0
    num_failed_extractions: int = 0
    failed_text_unit_ids: list[str] = []
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
        show_progress: bool = True,
        *,
        providers: Providers | None = None,
    ):
        super().__init__(config)
        self.providers = Providers.resolve(config, providers)
        self.claim_extraction_config = self.config.processing.claim_extraction
        self.ignore_errors = self.config.processing.ignore_errors
        self.show_progress = show_progress
        self.factory = self.providers.llm_factory
        self.batch_processor = BatchProcessor.from_processing(
            self.config.processing,
            is_transient_error=transient_error_predicate(self.factory),
        )

        robust_xml_output_parser = create_robust_xml_output_parser(
            model_purpose=ModelPurpose.INGESTION,
            factory=self.factory,
            enable_output_fixing=self.config.fixing.enabled,
            output_fixing_model_id=self.config.model_id(ModelRole.FIXING),
            min_output_tokens=ClaimExtractionPrompt.output_floor(self.config),
            output_tags=["claims"],
        )
        self.claim_extractor = setup_chain(
            model_purpose=ModelPurpose.INGESTION,
            factory=self.factory,
            model_id=self.config.model_id(ModelRole.CLAIM_EXTRACTION),
            prompt_class=ClaimExtractionPrompt,
            min_output_tokens=ClaimExtractionPrompt.output_floor(self.config),
            parser=robust_xml_output_parser,
            custom_prompts=self.config.custom_prompts,
        )

        self.stats: ClaimExtractionStats = ClaimExtractionStats()

    def extract_from_text_units(
        self, text_units: list[TextUnit], entities: list[Entity] | None = None
    ) -> tuple[list[Claim], ClaimExtractionStats]:
        """The claims of ``text_units``, merged across units (see
        :meth:`extract_units`)."""
        per_unit, stats = self.extract_units(text_units, entities)
        all_claims = [claim for claims in per_unit.values() for claim in claims]
        merged_claims = self.merge_claims(all_claims)
        stats.total_claims_extracted = len(merged_claims)
        logger.info(
            "Claim extraction completed: extracted %s claims, merged to %s unique claims",
            len(all_claims),
            len(merged_claims),
        )
        return merged_claims, stats

    def extract_units(
        self, text_units: list[TextUnit], entities: list[Entity] | None = None
    ) -> tuple[dict[str, list[Claim]], ClaimExtractionStats]:
        """Extract each text unit's claims.

        ``entities`` are listed in the prompt of each unit they cite. Returns
        the claims by text unit id, one entry (possibly empty) per unit whose
        extraction succeeded; the others are in the stats'
        ``failed_text_unit_ids``.
        """
        if not text_units:
            logger.warning("No text units provided for claim extraction")
            return {}, ClaimExtractionStats()

        all_entities = entities or []
        logger.info(
            "Starting claim extraction from %s text units with %s entities for relevance filtering",
            len(text_units),
            len(all_entities),
        )

        self.stats = ClaimExtractionStats(num_total_units=len(text_units))
        start_time = time.time()

        unit_to_input = self._prepare_extraction_inputs(text_units, all_entities)

        # Process ONLY units that have a prepared input, and align the batch
        # input list 1:1 with this filtered list. Otherwise prepare_inputs_func
        # would silently drop input-less units within a chunk, shrinking the
        # results list and misaligning the positional zip in
        # _process_extraction_results (claims attributed to the wrong unit).
        units_to_process = [u for u in text_units if u.id in unit_to_input]
        for unit in text_units:
            if unit.id not in unit_to_input:
                self._record_failure(unit)

        def prepare_inputs_for_chunk(
            chunk_items: list[TextUnit],
        ) -> list[dict[str, Any]]:
            return [unit_to_input[unit.id] for unit in chunk_items]

        try:
            extraction_results = self.batch_processor.execute_with_fallback(
                items_to_process=units_to_process,
                prepare_inputs_func=prepare_inputs_for_chunk,
                sequential_func=self.claim_extractor.invoke,
                task_name="Claim Extraction",
                show_progress=self.show_progress,
            )
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.error("Error during claim extraction: %s", e)
            for unit in units_to_process:
                self._record_failure(unit)
            return {}, self.stats

        per_unit = self._process_extraction_results(
            units_to_process, extraction_results
        )
        self.stats.total_claims_extracted = sum(len(c) for c in per_unit.values())
        self.stats.total_processing_time = time.time() - start_time
        self._log_completion_summary(self.stats)
        return per_unit, self.stats

    def _prepare_extraction_inputs(
        self, text_units: list[TextUnit], all_entities: list[Entity]
    ) -> dict[str, dict[str, Any]]:
        config_for_task = {
            "max_entities_per_prompt": self.claim_extraction_config.max_entities_per_prompt,
            "target_language": self.config.processing.translation.target_language.value,
        }

        # One lineage index for all units: each unit's prompt lists the
        # entities citing it, a lookup here rather than a scan of every entity
        # per unit in a worker that would be sent all of them.
        entities_by_unit = group_by_text_unit(all_entities)
        unit_to_input: dict[str, dict[str, Any]] = {}
        for unit in text_units:
            try:
                unit_to_input[unit.id] = prepare_claim_input(
                    unit, entities_by_unit.get(unit.id, []), config_for_task
                )
            except Exception as e:
                logger.error(
                    "Error preparing claim input for unit '%s': %s", unit.id, e
                )

        return unit_to_input

    def _process_extraction_results(
        self, text_units: list[TextUnit], extraction_results: list[Any]
    ) -> dict[str, list[Claim]]:
        per_unit: dict[str, list[Claim]] = {}

        # The batch processor returns one result per item (BATCH_ITEM_FAILED for
        # a failed one), so a length mismatch is a bug, not lost results.
        for text_unit, result in zip(text_units, extraction_results, strict=True):
            if result and result is not BATCH_ITEM_FAILED:
                try:
                    per_unit[text_unit.id] = self._parse_extraction_result(
                        result, text_unit
                    )
                    self.stats.num_successful_extractions += 1
                except Exception as e:
                    self._record_failure(text_unit)
                    logger.warning(
                        "Failed to parse extraction result for text unit '%s': %s",
                        text_unit.id,
                        e,
                    )
            else:
                self._record_failure(text_unit)
                logger.debug(
                    "No claim extraction result for text unit '%s'", text_unit.id
                )
        return per_unit

    def _record_failure(self, text_unit: TextUnit) -> None:
        self.stats.num_failed_extractions += 1
        self.stats.failed_text_unit_ids.append(text_unit.id)

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
    def _get_string_value(data: Any, *, join: bool = False) -> str:
        """Coerce an XML-parsed claim field to one stripped string.

        A repeated tag arrives as a list; like ``coerce_llm_text`` (#124),
        identity fields take the first non-empty element and ``join=True``
        (description-like fields) keeps every distinct element, newline-joined,
        instead of stringifying the list (``"['A', 'B']"``).
        """
        if isinstance(data, dict):
            if "#text" in data:
                return str(data["#text"]).strip()
            if len(data.values()) == 1:
                return ClaimExtractor._get_string_value(
                    list(data.values())[0], join=join
                )
            return str(data)
        if isinstance(data, list):
            return coerce_llm_text(
                [ClaimExtractor._get_string_value(item) for item in data], join=join
            )
        return coerce_llm_text(data)

    def _parse_claim_data(
        self, claim_data: dict[str, Any], text_unit: TextUnit
    ) -> Claim | None:
        try:
            subject_name = self._get_string_value(claim_data.get("subject"))
            object_name = self._get_string_value(claim_data.get("object"))
            claim_type = self._get_string_value(claim_data.get("claim_type"))

            if not all([subject_name, object_name, claim_type]):
                return None

            start_date = self._get_string_value(claim_data.get("start_date"))
            end_date = self._get_string_value(claim_data.get("end_date"))
            claim_id = self._generate_claim_id(
                subject_name, object_name, claim_type, start_date, end_date
            )
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
                start_date=start_date,
                end_date=end_date,
                description=self._get_string_value(
                    claim_data.get("description"), join=True
                ),
                description_embedding=None,
                text_unit_ids=[text_unit.id],
                source_text=self._get_string_value(
                    claim_data.get("source_text"), join=True
                ),
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
    def _generate_claim_id(
        subject_name: str,
        object_name: str,
        claim_type: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str:
        """Provisional claim id from the names as extracted, and the type.

        Only merges restatements within the extraction; ``ClaimResolver``
        replaces it with the id of the resolved entities
        (``resolved_claim_id``), which is the id that is stored. Endpoints use
        the same identity key as entity ids, so case/whitespace/quote variants
        of one name share an id. Known dates are part of the identity (see
        ``claim_identity_suffix``) so same-triple claims with different dates
        are not collapsed into one keeping a single date.
        """
        claim_id_content = (
            f"claim:{entity_key(subject_name)}:{entity_key(object_name)}:"
            f"{claim_type}{claim_identity_suffix(start_date, end_date)}".lower()
        )
        return generate_stable_id(claim_id_content)

    @staticmethod
    def merge_claims(claims: list[Claim]) -> list[Claim]:
        """Merge claim statements by id (see ``merge_claim_contributions``)."""
        return merge_claim_contributions(claims)

    @staticmethod
    def _log_completion_summary(stats: ClaimExtractionStats) -> None:
        if not stats:
            return

        logger.info(
            "Processing completed in %.2fs - "
            "Success rate: %.1f%% "
            "(%s/%s units processed)",
            stats.total_processing_time,
            stats.success_rate,
            stats.num_successful_extractions,
            stats.num_total_units,
        )

        if stats.num_failed_extractions > 0:
            logger.warning(
                "%s text units failed during extraction", stats.num_failed_extractions
            )
