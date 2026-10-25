# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from typing import Any

from pydantic import BaseModel, Field

from unified_kg_rag.adapters.llm.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from unified_kg_rag.adapters.llm.retry import (
    item_error_predicate,
    item_failed_indices,
    transient_error_predicate,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
from unified_kg_rag.domain.ingestion.contributions import (
    merge_entity_contributions,
    merge_relationship_contributions,
)
from unified_kg_rag.domain.ingestion.entity_grounding import is_grounded
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    ModelPurpose,
    ModelRole,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.domain.prompts import GraphExtractionPrompt
from unified_kg_rag.shared import DataProcessingError, get_logger
from unified_kg_rag.shared.utils import ensure_list, entity_key
from unified_kg_rag.shared.utils.langchain import BATCH_ITEM_FAILED, BatchProcessor

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
    failed_text_unit_ids: list[str] = Field(
        default_factory=list,
        description="Ids of the text units whose extraction failed",
    )
    item_failed_text_unit_ids: list[str] = Field(
        default_factory=list,
        description="The failed text units whose error was their own, not the "
        "model's (adapters.llm.retry.item_error_predicate): a request the "
        "model rejects for its content, or a response that does not parse or "
        "was cut off",
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
    relationships_ungrounded: int = Field(
        default=0,
        description="Number of relationships whose source_text span was not found "
        "in their source chunk (dropped or weight-penalized by the grounding guard)",
    )
    relationships_dropped_ungrounded_endpoint: int = Field(
        default=0,
        description="Number of relationships dropped because an endpoint entity "
        "was dropped by the grounding guard in the same chunk (otherwise the "
        "relationship would recreate the hallucinated entity as a stub)",
    )
    relationships_dropped_ungrounded_unlisted_endpoint: int = Field(
        default=0,
        description="Number of ungrounded relationships dropped in 'penalize' "
        "mode because their chunk did not extract one of their endpoints "
        "(keeping them would make that entity cite the chunk)",
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
        self,
        config: Config,
        *,
        providers: Providers | None = None,
    ) -> None:
        super().__init__(config)
        self.providers = Providers.resolve(config, providers)
        self.ignore_errors = self.config.processing.ignore_errors
        self.factory = self.providers.llm_factory
        self.batch_processor = BatchProcessor.from_processing(
            self.config.processing,
            is_transient_error=transient_error_predicate(self.factory),
        )
        self.is_item_error = item_error_predicate(self.factory)

        robust_xml_output_parser = create_robust_xml_output_parser(
            model_purpose=ModelPurpose.INGESTION,
            enable_output_fixing=self.config.fixing.enabled,
            output_fixing_model_id=self.config.model_id(ModelRole.FIXING),
            factory=self.factory,
            min_output_tokens=GraphExtractionPrompt.output_floor(self.config),
            output_tags=["entities", "relationships"],
        )
        self.graph_extractor = setup_chain(
            model_purpose=ModelPurpose.INGESTION,
            factory=self.factory,
            model_id=self.config.model_id(ModelRole.GRAPH_EXTRACTION),
            prompt_class=GraphExtractionPrompt,
            min_output_tokens=GraphExtractionPrompt.output_floor(self.config),
            parser=robust_xml_output_parser,
            custom_prompts=self.config.custom_prompts,
        )

        self.stats: ExtractionStats = ExtractionStats()

    def extract_units(
        self, text_units: list[TextUnit]
    ) -> tuple[list[TextUnitExtraction], ExtractionStats]:
        """Extract each text unit's record (its own entities and relationships).

        A unit whose extraction failed has no record; its id is in the
        stats' ``failed_text_unit_ids``.
        """
        start_time = time.time()

        if not text_units:
            logger.warning("No text units provided for graph extraction")
            return [], ExtractionStats()

        self.stats = ExtractionStats(num_total_units=len(text_units))
        logger.info("Starting graph extraction from %s text units", len(text_units))

        errors: dict[int, BaseException] = {}
        try:
            extraction_results = self.batch_processor.execute_with_fallback(
                items_to_process=text_units,
                prepare_inputs_func=self._prepare_extraction_inputs,
                sequential_func=self.graph_extractor.invoke,
                task_name="Graph Extraction",
                show_progress=self.show_progress,
                errors=errors,
            )
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.error("Error during graph extraction: %s", e)
            self.stats.num_failed_extractions = len(text_units)
            self.stats.failed_text_unit_ids = [unit.id for unit in text_units]
            return [], self.stats

        records = self._process_extraction_results(text_units, extraction_results)
        self.stats.item_failed_text_unit_ids.extend(
            text_units[idx].id
            for idx in item_failed_indices(
                errors,
                [unit.document_ids for unit in text_units],
                self.is_item_error,
                "Graph extraction",
            )
        )

        self.stats.total_entities_extracted = sum(len(r.entities) for r in records)
        self.stats.total_relationships_extracted = sum(
            len(r.relationships) for r in records
        )
        self.stats.total_processing_time = time.time() - start_time

        self._log_completion_summary(self.stats)

        return records, self.stats

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
        # Descriptions are written in the graph's language: the language the
        # text is translated to (equal to the source when translation is a
        # no-op), which is also the language queries are translated to.
        target_language = self.config.processing.translation.target_language.value

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
                        "target_language": target_language,
                    }
                )
            except Exception as e:
                failed_preparations += 1
                logger.error(
                    "Failed to prepare input for text unit '%s': %s",
                    text_unit.id,
                    str(e),
                )
                # Append a sentinel (empty input) rather than skipping, so the
                # returned inputs stay 1:1 with text_units. execute_with_fallback
                # returns results aligned to prepared inputs, and the downstream
                # `zip(text_units, results, strict=True)` would otherwise raise on
                # a length mismatch and crash the whole extraction stage (the same
                # latent bug fixed in the gleaner). The empty input yields an
                # empty extraction for that unit, which _process_extraction_results
                # already handles.
                inputs.append(
                    {
                        "input_text": "",
                        "max_entities_per_chunk": str(
                            graph_extraction_config.max_entities_per_chunk
                        ),
                        "max_relationships_per_chunk": str(
                            graph_extraction_config.max_relationships_per_chunk
                        ),
                        "entity_types": entity_types_block,
                        "target_language": target_language,
                    }
                )

        if failed_preparations > 0:
            logger.warning(
                "Failed to prepare inputs for %s text units", failed_preparations
            )

        return inputs

    def _process_extraction_results(
        self, text_units: list[TextUnit], extraction_results: list[Any]
    ) -> list[TextUnitExtraction]:
        records: list[TextUnitExtraction] = []
        for text_unit, result in zip(text_units, extraction_results, strict=True):
            if result is BATCH_ITEM_FAILED:
                self.stats.num_failed_extractions += 1
                self.stats.failed_text_unit_ids.append(text_unit.id)
                logger.warning("No extraction result for text unit '%s'", text_unit.id)
                continue
            try:
                records.append(self._parse_extraction_result(result, text_unit))
                self.stats.num_successful_extractions += 1
            except Exception as e:
                self.stats.num_failed_extractions += 1
                self.stats.failed_text_unit_ids.append(text_unit.id)
                self.stats.item_failed_text_unit_ids.append(text_unit.id)
                logger.error(
                    "Failed to parse extraction result for text unit '%s': %s",
                    text_unit.id,
                    str(e),
                )

        entities = [e for record in records for e in record.entities]
        if entities:
            self.stats.average_entity_confidence = sum(
                e.confidence or 1.0 for e in entities
            ) / len(entities)
        self.stats.confidence_threshold_applied = (
            self.extraction_config.entity_confidence_threshold
        )
        return records

    def _parse_extraction_result(
        self,
        result: dict[str, Any],
        text_unit: TextUnit,
    ) -> TextUnitExtraction:
        """Parse one chain output into the unit's record; raise
        ``DataProcessingError`` if malformed.

        The output must carry an ``entities`` section. ``relationships`` may be
        absent: the XML parser drops an empty section, so a unit whose entities
        are unrelated has none. A name listed twice is one entity of the unit
        (and a relationship stated twice one relationship, its strengths
        summed), as in the merge across units.
        """
        entities: list[Entity] = []
        relationships: list[Relationship] = []

        if not isinstance(result, dict):
            raise DataProcessingError(
                f"extraction output is {type(result).__name__}, not a mapping"
            )
        raw_entities = result.get("entities")
        raw_relationships = result.get("relationships", {})
        if not isinstance(raw_entities, dict | list) or not isinstance(
            raw_relationships, dict | list
        ):
            raise DataProcessingError(
                "extraction output lacks well-formed entities/relationships "
                f"sections (keys: {sorted(result)})"
            )

        entities_data = ensure_list(raw_entities, inner_key="entity")
        for entity_data in entities_data:
            if entity := self.parse_entity_data(entity_data, text_unit):
                entities.append(entity)

        entities, dropped_keys = self._apply_entity_grounding(entities, text_unit)
        entities, low_confidence_keys = self._apply_confidence_threshold(
            entities, text_unit
        )
        record = TextUnitExtraction(text_unit_id=text_unit.id)
        for key in sorted(dropped_keys):
            record.reject(key, "ungrounded")
        for key in sorted(low_confidence_keys):
            record.reject(key, "low_confidence")

        entity_name_to_id = self.build_entity_key_index(entities)
        relationships_data = ensure_list(raw_relationships, inner_key="relationship")
        for rel_data in relationships_data:
            if relationship := self.parse_relationship_data(
                rel_data, text_unit, entity_name_to_id
            ):
                relationships.append(relationship)

        kept_relationships = self._drop_relationships_to_dropped_entities(
            relationships, dropped_keys, text_unit, "ungrounded"
        )
        self.stats.relationships_dropped_ungrounded_endpoint += len(
            relationships
        ) - len(kept_relationships)
        relationships = kept_relationships
        kept_relationships = self._drop_relationships_to_dropped_entities(
            relationships, low_confidence_keys, text_unit, "low confidence"
        )
        self.stats.relationships_filtered_by_confidence += len(relationships) - len(
            kept_relationships
        )
        relationships = kept_relationships
        relationships = self._apply_relationship_grounding(
            relationships, text_unit, {e.id for e in entities}
        )

        record.entities = merge_entity_contributions(entities)
        record.relationships = merge_relationship_contributions(
            [rel for rel in relationships if rel.source_id != rel.target_id]
        )
        return record

    def _apply_confidence_threshold(
        self, entities: list[Entity], text_unit: TextUnit
    ) -> tuple[list[Entity], set[str]]:
        """Drop the chunk's entities below ``entity_confidence_threshold``.

        Per chunk, after grounding (so a ``penalize``d confidence counts), as
        gleaning applies it to the entities it adds. Thresholding the merged
        entity instead would test the best score over the chunks that share
        the batch, so a full build and an incremental run would keep different
        entities. A dropped entity's relationships in the chunk go with it,
        else the stub step would recreate it there.

        Returns the kept entities and the ``entity_key`` of every entity
        dropped (excluding keys with a kept copy in the chunk).
        """
        threshold = self.extraction_config.entity_confidence_threshold
        if threshold <= 0.0:
            return entities, set()
        kept: list[Entity] = []
        dropped_keys: set[str] = set()
        for entity in entities:
            confidence = entity.confidence if entity.confidence is not None else 1.0
            if confidence >= threshold:
                kept.append(entity)
                continue
            self.stats.entities_filtered_by_confidence += 1
            dropped_keys.add(entity_key(entity.name))
            logger.debug(
                "Filtered entity '%s' with confidence %.2f (threshold: %s, chunk '%s')",
                entity.short_id,
                confidence,
                threshold,
                text_unit.short_id,
            )
        dropped_keys -= {entity_key(e.name) for e in kept}
        return kept, dropped_keys

    def _apply_entity_grounding(
        self, entities: list[Entity], text_unit: TextUnit
    ) -> tuple[list[Entity], set[str]]:
        """Drop or penalize entities not grounded in their source chunk.

        Per-chunk (before merge), so each entity is checked against the exact
        text it was extracted from. The verbatim evidence span lives in the
        reserved ``_source_text`` attribute (stripped here so it is not
        persisted). No-op when grounding is disabled.

        Returns the kept entities and the ``entity_key`` of every entity dropped
        in this chunk (excluding keys that also have a kept, grounded copy), so
        relationships naming a dropped entity can be removed before stub
        materialization would recreate it.
        """
        grounding = self.extraction_config.entity_grounding
        chunk_text = self.get_text_for_processing(text_unit)

        kept: list[Entity] = []
        dropped_keys: set[str] = set()
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
                dropped_keys.add(entity_key(entity.name))
                logger.info(
                    "Dropping ungrounded entity '%s' — source_text not found in "
                    "chunk '%s' (likely hallucinated)",
                    entity.short_id,
                    text_unit.short_id,
                )
        dropped_keys -= {entity_key(e.name) for e in kept}
        return kept, dropped_keys

    def _drop_relationships_to_dropped_entities(
        self,
        relationships: list[Relationship],
        dropped_keys: set[str],
        text_unit: TextUnit,
        reason: str,
    ) -> list[Relationship]:
        """Remove relationships whose endpoint the chunk dropped (as ``reason``,
        for the log); the caller counts the removed ones.

        Without this, ``_materialize_relationship_endpoints`` would recreate the
        dropped (ungrounded or low-confidence) entity as a description-less
        stub and undo the guard.
        """
        if not dropped_keys:
            return relationships
        kept: list[Relationship] = []
        for rel in relationships:
            if (
                entity_key(rel.source_name or "") in dropped_keys
                or entity_key(rel.target_name or "") in dropped_keys
            ):
                logger.info(
                    "Dropping relationship '%s' — endpoint entity was dropped "
                    "as %s in chunk '%s'",
                    rel.short_id,
                    reason,
                    text_unit.short_id,
                )
                continue
            kept.append(rel)
        return kept

    def _apply_relationship_grounding(
        self,
        relationships: list[Relationship],
        text_unit: TextUnit,
        chunk_entity_ids: set[str],
    ) -> list[Relationship]:
        """Drop or weight-penalize relationships not grounded in their chunk.

        Mirrors entity grounding: the reserved ``_source_text`` attribute holds
        the verbatim evidence span (stripped here). In ``penalize`` mode the
        relationship weight (not confidence — relationships have no confidence)
        is scaled by ``penalty_factor``. No-op when grounding is disabled.

        ``penalize`` keeps an ungrounded relationship only when the chunk
        extracted both endpoints (``chunk_entity_ids``), which therefore cite
        the chunk already. Every kept relationship makes its endpoints cite
        its chunk (see :func:`cite_relationship_endpoints`), so keeping one
        that names an entity the chunk did not list would make that entity,
        possibly well grounded elsewhere, cite a chunk with no evidence for it.
        Exempting ungrounded edges from the citation instead would make the
        entity's text units depend on the batch (a stub when no other chunk
        lists it, nothing otherwise), which incremental removal cannot
        reproduce, so such a relationship is dropped.
        """
        grounding = self.extraction_config.entity_grounding
        chunk_text = self.get_text_for_processing(text_unit)

        kept: list[Relationship] = []
        for rel in relationships:
            source_text = (rel.attributes or {}).pop("_source_text", None)
            grounded = (not grounding.enabled) or is_grounded(
                source_text,
                chunk_text,
                min_span_tokens=grounding.min_span_tokens,
                min_overlap_ratio=grounding.min_overlap_ratio,
            )

            if grounded:
                kept.append(rel)
                continue

            self.stats.relationships_ungrounded += 1
            if (
                grounding.action == "penalize"
                and not {
                    rel.source_id,
                    rel.target_id,
                }
                <= chunk_entity_ids
            ):
                self.stats.relationships_dropped_ungrounded_unlisted_endpoint += 1
                logger.info(
                    "Dropping ungrounded relationship '%s' — it names an "
                    "entity chunk '%s' did not extract",
                    rel.short_id,
                    text_unit.short_id,
                )
            elif grounding.action == "penalize":
                rel.weight = (rel.weight or 1.0) * grounding.penalty_factor
                kept.append(rel)
                logger.debug(
                    "Ungrounded relationship '%s -> %s' penalized to weight %.2f "
                    "(chunk '%s')",
                    rel.source_name,
                    rel.target_name,
                    rel.weight,
                    text_unit.short_id,
                )
            else:  # drop
                logger.info(
                    "Dropping ungrounded relationship '%s' — source_text "
                    "not found in chunk '%s' (likely hallucinated)",
                    rel.short_id,
                    text_unit.short_id,
                )
        return kept

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

        if stats.entities_ungrounded or stats.relationships_ungrounded:
            logger.info(
                "Grounding guard - Entities ungrounded: %s, Relationships "
                "ungrounded: %s, Relationships dropped with an ungrounded "
                "endpoint: %s, Ungrounded relationships dropped naming an "
                "unlisted entity: %s",
                stats.entities_ungrounded,
                stats.relationships_ungrounded,
                stats.relationships_dropped_ungrounded_endpoint,
                stats.relationships_dropped_ungrounded_unlisted_endpoint,
            )

        if stats.num_failed_extractions > 0:
            logger.warning(
                "Failed to extract from %s text units", stats.num_failed_extractions
            )
