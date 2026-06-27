# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from collections.abc import Callable
from typing import Any

import boto3
from pydantic import BaseModel, Field

from unified_kg_rag.adapters.aws import BedrockLanguageModelFactory
from unified_kg_rag.adapters.aws.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.entity_grounding import is_grounded
from unified_kg_rag.domain.models import Config, Entity, Relationship, TextUnit
from unified_kg_rag.domain.prompts import GraphExtractionPrompt
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import (
    BatchProcessor,
    ensure_list,
)

logger = get_logger(__name__)


class ExtractionStats(BaseModel):
    num_total_units: int = Field(
        default=0,
        description="Total number of text units processed for graph extraction",
    )
    num_successful_extractions: int = Field(
        default=0, description="Number of text units that were successfully extracted"
    )
    num_failed_extractions: int = Field(
        default=0, description="Number of text units that encountered extraction errors"
    )
    total_entities_extracted: int = Field(
        default=0, description="Total number of entities extracted from all text units"
    )
    total_relationships_extracted: int = Field(
        default=0,
        description="Total number of relationships extracted from all text units",
    )
    total_processing_time: float = Field(
        default=0.0, description="Total time spent processing extractions (in seconds)"
    )
    entities_filtered_by_confidence: int = Field(
        default=0,
        description="Number of entities filtered out due to low confidence score",
    )
    relationships_filtered_by_confidence: int = Field(
        default=0,
        description="Number of relationships filtered out due to entity confidence filtering",
    )
    average_entity_confidence: float = Field(
        default=0.0,
        description="Average confidence score of extracted entities",
    )
    confidence_threshold_applied: float = Field(
        default=0.0,
        description="Confidence threshold value used for filtering",
    )
    entities_ungrounded: int = Field(
        default=0,
        description="Number of entities whose source_text span was not found in "
        "their source chunk (dropped or confidence-penalized by the grounding guard)",
    )

    @property
    def processed_unit_count(self) -> int:
        return self.num_successful_extractions + self.num_failed_extractions

    @property
    def average_processing_time(self) -> float:
        if self.processed_unit_count == 0:
            return 0.0
        return self.total_processing_time / self.processed_unit_count

    @property
    def success_rate(self) -> float:
        if self.processed_unit_count == 0:
            return 0.0
        return (self.num_successful_extractions / self.processed_unit_count) * 100


class GraphExtractor(BaseProcessor):
    def __init__(
        self, config: Config, boto_session: boto3.Session | None = None
    ) -> None:
        super().__init__(config)
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name
        )
        self.ignore_errors = self.config.processing.ignore_errors
        self.factory = BedrockLanguageModelFactory(
            config=self.config,
            boto_session=self.boto_session,
            region_name=self.config.aws.bedrock.region_name,
        )
        self.batch_processor = BatchProcessor()

        robust_xml_output_parser = create_robust_xml_output_parser(
            enable_output_fixing=self.config.fixing.enabled,
            output_fixing_model_id=self.config.fixing.fixing_model_id,
            factory=self.factory,
        )
        self.graph_extractor = setup_chain(
            factory=self.factory,
            model_id=self.extraction_config.extraction_model_id,
            prompt_class=GraphExtractionPrompt,
            parser=robust_xml_output_parser,
            custom_prompts=self.config.custom_prompts,
        )

        self.stats: ExtractionStats = ExtractionStats()

    def extract_from_text_units(
        self, text_units: list[TextUnit]
    ) -> tuple[list[Entity], list[Relationship], ExtractionStats]:
        start_time = time.time()

        if not text_units:
            logger.warning("No text units provided for graph extraction")
            return [], [], ExtractionStats()

        self.stats = ExtractionStats(num_total_units=len(text_units))
        logger.info("Starting graph extraction from %s text units", len(text_units))

        try:
            extraction_results = self.batch_processor.execute_with_fallback(
                items_to_process=text_units,
                prepare_inputs_func=self._prepare_extraction_inputs,
                batch_func=self.graph_extractor.batch,
                sequential_func=self.graph_extractor.invoke,
                task_name="Graph Extraction",
                run_config=self.config.processing.model_dump(),
                show_progress=self.show_progress,
            )
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.error("Error during graph extraction: %s", e)
            return [], [], ExtractionStats()

        all_entities, all_relationships = self._process_extraction_results(
            text_units, extraction_results
        )

        self.stats.total_entities_extracted = len(all_entities)
        self.stats.total_relationships_extracted = len(all_relationships)
        self.stats.total_processing_time = time.time() - start_time

        self._log_completion_summary(self.stats)

        return all_entities, all_relationships, self.stats

    @staticmethod
    def _format_entity_types(entity_types: list[str]) -> str:
        """Render config entity types into the prompt's category block.

        Each item is rendered as ``- **LABEL**: description`` (the description
        after the first ':' is optional). An empty list lets the model choose
        any relevant types — the prompt slot then says so explicitly.
        """
        if not entity_types:
            return "Use any entity types relevant to the document's domain."
        lines = []
        for item in entity_types:
            label, sep, desc = item.partition(":")
            label = label.strip()
            if not label:
                continue
            lines.append(f"- **{label}**:{desc.rstrip()}" if sep else f"- **{label}**")
        return "\n".join(lines)

    def _prepare_extraction_inputs(
        self, text_units: list[TextUnit]
    ) -> list[dict[str, str]]:
        inputs = []
        graph_extraction_config = self.extraction_config
        failed_preparations = 0
        entity_types_block = self._format_entity_types(
            graph_extraction_config.entity_types
        )

        for text_unit in text_units:
            try:
                inputs.append(
                    {
                        "input_text": self.get_text_for_processing(text_unit),
                        "max_entities_per_chunk": str(
                            graph_extraction_config.max_entities_per_chunk
                        ),
                        "max_relationships_per_chunk": str(
                            graph_extraction_config.max_relationships_per_chunk
                        ),
                        "entity_types": entity_types_block,
                    }
                )
            except Exception as e:
                failed_preparations += 1
                logger.error(
                    "Failed to prepare input for text unit '%s': %s",
                    text_unit.id,
                    str(e),
                )

        if failed_preparations > 0:
            logger.warning(
                "Failed to prepare inputs for %s text units", failed_preparations
            )

        return inputs

    def _process_extraction_results(
        self, text_units: list[TextUnit], extraction_results: list[Any]
    ) -> tuple[list[Entity], list[Relationship]]:
        all_entities = []
        all_relationships = []

        for text_unit, result in zip(text_units, extraction_results, strict=True):
            if result:
                try:
                    entities, relationships = self._parse_extraction_result(
                        result, text_unit
                    )
                    all_entities.extend(entities)
                    all_relationships.extend(relationships)
                    self.stats.num_successful_extractions += 1
                except Exception as e:
                    self.stats.num_failed_extractions += 1
                    logger.error(
                        "Failed to parse extraction result for text unit '%s': %s",
                        text_unit.id,
                        str(e),
                    )
            else:
                self.stats.num_failed_extractions += 1
                logger.warning("No extraction result for text unit '%s'", text_unit.id)

        original_entities_count = len(all_entities)
        original_relationships_count = len(all_relationships)

        all_entities = self._merge_entities(all_entities)
        all_relationships = self._merge_relationships(all_relationships)

        # Materialize entities referenced only by a relationship endpoint (the
        # LLM mentioned them in a relation but did not list them as an entity),
        # so those relationships are not later skipped as orphans. MS GraphRAG
        # likewise lets relationships introduce entities.
        all_entities = self._materialize_relationship_endpoints(
            all_entities, all_relationships
        )

        all_entities, filtered_count = self._filter_entities_by_confidence(all_entities)
        self.stats.entities_filtered_by_confidence = filtered_count

        if filtered_count > 0:
            all_relationships, rel_filtered_count = self._filter_orphan_relationships(
                all_relationships, all_entities
            )
            self.stats.relationships_filtered_by_confidence = rel_filtered_count

        if all_entities:
            self.stats.average_entity_confidence = sum(
                e.confidence or 1.0 for e in all_entities
            ) / len(all_entities)

        self.stats.confidence_threshold_applied = (
            self.extraction_config.entity_confidence_threshold
        )

        logger.info(
            "Entity and relationship processing completed - %s -> %s entities (filtered: %s), %s -> %s relationships",
            original_entities_count,
            len(all_entities),
            self.stats.entities_filtered_by_confidence,
            original_relationships_count,
            len(all_relationships),
        )
        return all_entities, all_relationships

    def _parse_extraction_result(
        self,
        result: dict[str, Any],
        text_unit: TextUnit,
    ) -> tuple[list[Entity], list[Relationship]]:
        entities: list[Entity] = []
        relationships: list[Relationship] = []

        if not isinstance(result, dict) or (
            "entities" not in result or "relationships" not in result
        ):
            logger.warning(
                "Invalid result type for text unit '%s': '%s'",
                text_unit.id,
                type(result),
            )
            return entities, relationships

        entities_data = ensure_list(result.get("entities"), inner_key="entity")
        for entity_data in entities_data:
            if entity := self.parse_entity_data(entity_data, text_unit):
                entities.append(entity)

        entities = self._apply_entity_grounding(entities, text_unit)

        entity_name_to_id = {entity.name: entity.id for entity in entities}
        relationships_data = ensure_list(
            result.get("relationships"), inner_key="relationship"
        )
        for rel_data in relationships_data:
            if relationship := self.parse_relationship_data(
                rel_data, text_unit, entity_name_to_id
            ):
                relationships.append(relationship)

        return entities, relationships

    def _merge_entities(self, entities: list[Entity]) -> list[Entity]:
        field_mergers: dict[str, Callable[[Any, Any], Any]] = {
            "description": self._merge_description,
            "text_unit_ids": lambda current, new: list(
                set(
                    (current if isinstance(current, list) else [])
                    + (new if isinstance(new, list) else [])
                )
            ),
            # Confidence should be monotonic in evidence: an entity extracted
            # confidently many times should be reinforced, not diluted toward a
            # running mean (and made order-dependent). Take the max.
            "confidence": lambda current, new: max(
                current if current is not None else 0.0,
                new if new is not None else 0.0,
            ),
        }

        merged = self._merge_items(
            items=entities,
            item_name="Entity",
            field_mergers=field_mergers,
            frequency_fields=["type"],
            log_message_formatter=lambda e: f"Entity '{e.name}' merged {{count}} instances",
        )
        # Set frequency from text-unit support so full-build entities carry a
        # ranking signal (previously only the incremental merger set frequency,
        # leaving it None for full runs and degrading report/search ranking).
        for entity in merged:
            entity.frequency = len(entity.text_unit_ids or [])
        return merged

    def _merge_relationships(
        self, relationships: list[Relationship]
    ) -> list[Relationship]:
        field_mergers: dict[str, Callable[[Any, Any], Any]] = {
            "weight": lambda current, new: (
                current if isinstance(current, (int | float)) else 0.0
            )
            + (new if isinstance(new, (int | float)) else 0.0),
            "description": self._merge_description,
            "text_unit_ids": lambda current, new: list(
                set(
                    (current if isinstance(current, list) else [])
                    + (new if isinstance(new, list) else [])
                )
            ),
        }

        return self._merge_items(
            items=relationships,
            item_name="Relationship",
            field_mergers=field_mergers,
            log_message_formatter=lambda r: (
                f"Relationship '{r.source_name}' -> '{r.target_name}' "
                f"(type: '{r.type}') merged {{count}} instances"
            ),
        )

    def _filter_entities_by_confidence(
        self, entities: list[Entity]
    ) -> tuple[list[Entity], int]:
        threshold = self.extraction_config.entity_confidence_threshold

        if threshold <= 0.0:
            return entities, 0

        filtered_entities = []
        removed_count = 0

        for entity in entities:
            confidence = entity.confidence if entity.confidence is not None else 1.0
            if confidence >= threshold:
                filtered_entities.append(entity)
            else:
                removed_count += 1
                logger.debug(
                    "Filtered entity '%s' with confidence %.2f (threshold: %s)",
                    entity.name,
                    confidence,
                    threshold,
                )

        if removed_count > 0:
            logger.info(
                "Confidence filtering removed %s entities (threshold: %s)",
                removed_count,
                threshold,
            )

        return filtered_entities, removed_count

    def _apply_entity_grounding(
        self, entities: list[Entity], text_unit: TextUnit
    ) -> list[Entity]:
        """Drop or penalize entities not grounded in their source chunk.

        Per-chunk (before merge), so each entity is checked against the exact
        text it was extracted from. The verbatim evidence span lives in the
        reserved ``_source_text`` attribute (stripped here so it is not
        persisted). No-op when grounding is disabled.
        """
        grounding = self.extraction_config.entity_grounding
        chunk_text = self.get_text_for_processing(text_unit)

        kept: list[Entity] = []
        for entity in entities:
            source_text = (entity.attributes or {}).pop("_source_text", None)
            grounded = (not grounding.enabled) or is_grounded(
                source_text,
                chunk_text,
                min_span_tokens=grounding.min_span_tokens,
                min_overlap_ratio=grounding.min_overlap_ratio,
            )

            if grounded:
                kept.append(entity)
                continue

            # Ungrounded.
            self.stats.entities_ungrounded += 1
            if grounding.action == "penalize":
                base = entity.confidence if entity.confidence is not None else 1.0
                entity.confidence = base * grounding.penalty_factor
                kept.append(entity)
                logger.debug(
                    "Ungrounded entity '%s' penalized to confidence %.2f (chunk '%s')",
                    entity.name,
                    entity.confidence,
                    text_unit.short_id,
                )
            else:  # drop
                logger.info(
                    "Dropping ungrounded entity '%s' — source_text not found in "
                    "chunk '%s' (likely hallucinated)",
                    entity.name,
                    text_unit.short_id,
                )
        return kept

    def _materialize_relationship_endpoints(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> list[Entity]:
        """Create stub entities for relationship endpoints not already extracted.

        Endpoint ids are name-derived, so when the LLM references an entity only
        inside a relationship (or it was extracted in another chunk) we add a
        minimal entity for it rather than letting graph_builder drop the edge.
        """
        existing_ids = {e.id for e in entities}
        stubs: dict[str, Entity] = {}
        for rel in relationships:
            for ent_id, name in (
                (rel.source_id, rel.source_name),
                (rel.target_id, rel.target_name),
            ):
                if not ent_id or ent_id in existing_ids or ent_id in stubs or not name:
                    continue
                stubs[ent_id] = Entity.model_validate(
                    {
                        "id": ent_id,
                        "name": name,
                        "text_unit_ids": list(rel.text_unit_ids or []),
                    }
                )
        if stubs:
            logger.info(
                "Materialized %s entities referenced only by relationships",
                len(stubs),
            )
        return entities + list(stubs.values())

    def _filter_orphan_relationships(
        self,
        relationships: list[Relationship],
        valid_entities: list[Entity],
    ) -> tuple[list[Relationship], int]:
        valid_entity_ids = {entity.id for entity in valid_entities}

        filtered_relationships = []
        removed_count = 0

        for rel in relationships:
            if rel.source_id in valid_entity_ids and rel.target_id in valid_entity_ids:
                filtered_relationships.append(rel)
            else:
                removed_count += 1
                logger.debug(
                    "Filtered orphan relationship '%s' -> '%s'",
                    rel.source_name,
                    rel.target_name,
                )

        if removed_count > 0:
            logger.info(
                "Filtered %s orphan relationships (referenced entities were filtered by confidence)",
                removed_count,
            )

        return filtered_relationships, removed_count

    @staticmethod
    def _log_completion_summary(stats: ExtractionStats) -> None:
        if not stats:
            return

        logger.info(
            "Graph extraction completed - "
            "Time: %.2fs, "
            "Success rate: %.1f%% "
            "(%s/%s), "
            "Entities: %s, "
            "Relationships: %s",
            stats.total_processing_time,
            stats.success_rate,
            stats.num_successful_extractions,
            stats.num_total_units,
            stats.total_entities_extracted,
            stats.total_relationships_extracted,
        )

        if stats.entities_filtered_by_confidence > 0:
            logger.info(
                "Confidence filtering - "
                "Threshold: %.2f, "
                "Entities filtered: %s, "
                "Relationships filtered: %s, "
                "Average confidence: %.2f",
                stats.confidence_threshold_applied,
                stats.entities_filtered_by_confidence,
                stats.relationships_filtered_by_confidence,
                stats.average_entity_confidence,
            )

        if stats.num_failed_extractions > 0:
            logger.warning(
                "Failed to extract from %s text units", stats.num_failed_extractions
            )
