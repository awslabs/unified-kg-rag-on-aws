# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from functools import partial
from typing import Any

import boto3
from pydantic import BaseModel, Field
from tqdm import tqdm

from unified_kg_rag.adapters.aws.bedrock_retry import is_transient_bedrock_error
from unified_kg_rag.adapters.aws.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.ingestion.base_processor import (
    BaseProcessor,
    check_entity_relevance_task,
    check_relationship_relevance_task,
)
from unified_kg_rag.domain.ingestion.entity_grounding import is_grounded
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    ModelPurpose,
    Relationship,
    TextUnit,
)
from unified_kg_rag.domain.prompts import GraphRefinementPrompt
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import (
    clean_display_name,
    default_max_workers,
    ensure_list,
    entity_key,
)
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor
from unified_kg_rag.shared.utils.langchain import BATCH_ITEM_FAILED, BatchProcessor

logger = get_logger(__name__)


def format_entities_with_limit_task(entities: list[Entity], max_entities: int) -> str:
    if len(entities) <= max_entities:
        return "\n".join(e.name for e in entities)

    sorted_entities = sorted(
        entities,
        key=lambda e: (
            bool(e.description and e.description.strip()),
            len(e.text_unit_ids or []),
        ),
        reverse=True,
    )
    selected_entities = sorted_entities[:max_entities]
    entity_list = "\n".join(e.name for e in selected_entities)
    if len(entities) > max_entities:
        entity_list += f"\n... and {len(entities) - max_entities} more entities"
    return entity_list


def format_relationships_with_limit_task(
    relationships: list[Relationship], max_relationships: int
) -> str:
    if len(relationships) <= max_relationships:
        return "\n".join(
            f"'{r.source_name}' -> '{r.target_name}' (type: '{r.type}')"
            for r in relationships
        )

    sorted_relationships = sorted(
        relationships,
        key=lambda r: (
            r.weight or 0.0,
            bool(r.description and r.description.strip()),
        ),
        reverse=True,
    )
    selected_relationships = sorted_relationships[:max_relationships]
    rel_list = "\n".join(
        f"'{r.source_name}' -> '{r.target_name}' (type: '{r.type}')"
        for r in selected_relationships
    )
    if len(relationships) > max_relationships:
        rel_list += (
            f"\n... and {len(relationships) - max_relationships} more relationships"
        )
    return rel_list


def prepare_input_task(
    unit: TextUnit,
    all_entities: list[Entity],
    all_relationships: list[Relationship],
    config: dict[str, Any],
) -> dict[str, Any]:
    if unit.translated_texts:
        target_language = config.get("target_language", "en")
        unit_text = unit.translated_texts.get(target_language, unit.text or "")
    else:
        unit_text = unit.text or ""

    relevant_entities = []
    if all_entities:
        for entity in all_entities:
            _, is_relevant = check_entity_relevance_task(entity, unit.id)
            if is_relevant:
                relevant_entities.append(entity)

    relevant_relationships = []
    if all_relationships:
        for rel in all_relationships:
            _, is_relevant = check_relationship_relevance_task(rel, unit.id)
            if is_relevant:
                relevant_relationships.append(rel)

    entities_str = format_entities_with_limit_task(
        relevant_entities, config["max_entities_per_prompt"]
    )
    relationships_str = format_relationships_with_limit_task(
        relevant_relationships, config["max_relationships_per_prompt"]
    )

    return {
        "text": unit_text,
        "entities": entities_str,
        "relationships": relationships_str,
    }


class GleaningRound(BaseModel):
    round_number: int
    # Text units sent to the model this round: one refinement call each.
    units_gleaned: int
    # Text units whose answer added a new entity or relationship; only these
    # are gleaned again in the next round.
    units_gained: int
    entities_before: int
    relationships_before: int
    entities_added: int
    relationships_added: int
    processing_time: float


class GleaningStats(BaseModel):
    total_rounds: int = 0
    total_entities_added: int = 0
    total_relationships_added: int = 0
    total_processing_time: float = 0.0
    # Unit refinements that failed (input prep, LLM call, or whole batch),
    # summed over rounds; such units simply gain nothing from gleaning.
    num_failed_units: int = 0
    # Distinct ids of the units with at least one failed refinement.
    failed_text_unit_ids: list[str] = Field(default_factory=list)
    rounds: list[GleaningRound] = Field(default_factory=list)

    @property
    def total_refinement_calls(self) -> int:
        return sum(r.units_gleaned for r in self.rounds)


class GraphGleaner(BaseProcessor):
    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        max_workers: int | None = None,
        use_process_pool: bool = True,
        show_progress: bool = True,
        *,
        providers: Providers | None = None,
    ) -> None:
        super().__init__(config)
        self.providers = Providers.resolve(config, providers, boto_session)
        self.boto_session = self.providers.boto_session
        self.gleaning_config = self.config.processing.gleaning
        self.ignore_errors = self.config.processing.ignore_errors
        self.max_workers = max_workers or default_max_workers()
        self.use_process_pool = use_process_pool
        self.show_progress = show_progress
        self._failed_unit_ids: list[str] = []

        self.factory = self.providers.llm_factory
        self.batch_processor = BatchProcessor(
            is_transient_error=is_transient_bedrock_error
        )

        self.max_entities_per_prompt = self.gleaning_config.max_entities_per_prompt
        self.max_relationships_per_prompt = (
            self.gleaning_config.max_relationships_per_prompt
        )

        robust_xml_output_parser = create_robust_xml_output_parser(
            model_purpose=ModelPurpose.INGESTION,
            factory=self.factory,
            enable_output_fixing=self.config.fixing.enabled,
            output_fixing_model_id=self.config.fixing.fixing_model_id,
            min_output_tokens=GraphRefinementPrompt.min_output_tokens,
        )
        self.graph_refiner = setup_chain(
            model_purpose=ModelPurpose.INGESTION,
            factory=self.factory,
            model_id=self.gleaning_config.graph_refinement_model_id,
            prompt_class=GraphRefinementPrompt,
            parser=robust_xml_output_parser,
            custom_prompts=self.config.custom_prompts,
        )

    def glean_graph(
        self,
        text_units: list[TextUnit],
        initial_entities: list[Entity],
        initial_relationships: list[Relationship],
    ) -> tuple[list[Entity], list[Relationship], GleaningStats]:
        """Glean each text unit for at most ``max_rounds`` rounds.

        Round 1 sends every unit. Each later round re-sends only the units
        whose previous answer added a new entity or relationship. A unit stops
        once its answer adds nothing new: an empty ``<identified_issues>`` is
        the model saying nothing more is missing (the answer MS GraphRAG gets
        from its separate Y/N loop prompt), and items that merge into existing
        ones or fail grounding are not new.
        """
        start_time = time.time()
        current_entities = initial_entities.copy()
        current_relationships = initial_relationships.copy()
        stats = GleaningStats()
        self._failed_unit_ids = []
        max_rounds = self.gleaning_config.max_rounds

        logger.info(
            "Starting graph gleaning from %s text units, %s entities, "
            "%s relationships",
            len(text_units),
            len(current_entities),
            len(current_relationships),
        )

        units_to_glean = list(text_units)
        for round_num in range(1, max_rounds + 1):
            if not units_to_glean:
                break
            logger.info(
                "Starting gleaning round %s/%s (%s text units)",
                round_num,
                max_rounds,
                len(units_to_glean),
            )

            current_entities, current_relationships, round_info, gained = (
                self._perform_gleaning_round(
                    units_to_glean,
                    current_entities,
                    current_relationships,
                    round_num,
                )
            )
            stats.rounds.append(round_info)
            stats.total_entities_added += round_info.entities_added
            stats.total_relationships_added += round_info.relationships_added

            logger.info(
                "Round %s completed: +%s entities, +%s relationships, "
                "%s/%s text units gained",
                round_num,
                round_info.entities_added,
                round_info.relationships_added,
                round_info.units_gained,
                round_info.units_gleaned,
            )
            units_to_glean = [unit for unit in units_to_glean if unit.id in gained]

        stats.total_rounds = len(stats.rounds)
        stats.num_failed_units = len(self._failed_unit_ids)
        stats.failed_text_unit_ids = sorted(set(self._failed_unit_ids))
        stats.total_processing_time = time.time() - start_time

        self._log_completion_summary(stats, units_still_gaining=len(units_to_glean))

        return current_entities, current_relationships, stats

    @staticmethod
    def _relationship_key(rel: Relationship) -> tuple[str, str, str]:
        return (rel.source_id, rel.target_id, rel.type.lower() if rel.type else "")

    def _perform_gleaning_round(
        self,
        text_units: list[TextUnit],
        current_entities: list[Entity],
        current_relationships: list[Relationship],
        round_num: int,
    ) -> tuple[list[Entity], list[Relationship], GleaningRound, set[str]]:
        """Send each unit to the model once and merge the answers into the graph.

        Returns:
            The merged entities and relationships, the round's stats, and the
            ids of the units whose answer added a new entity or relationship.
        """
        round_start_time = time.time()
        entities_before = len(current_entities)
        relationships_before = len(current_relationships)
        # Taken before the model call, since a correction renames in place.
        known_entity_keys = {entity_key(e.name) for e in current_entities}
        known_relationship_keys = {
            self._relationship_key(r) for r in current_relationships
        }

        new_entities, new_relationships = self._perform_llm_refinement(
            text_units, current_entities, current_relationships
        )

        logger.debug(
            "Round %s: LLM refinement produced %s entities and %s relationships",
            round_num,
            len(new_entities),
            len(new_relationships),
        )

        merged_entities, entity_id_map = self._merge_duplicate_entities(
            current_entities + new_entities
        )
        merged_relationships = self._update_relationships_after_merge(
            current_relationships + new_relationships,
            {e.id for e in merged_entities},
            entity_id_map,
        )
        # Runs after every correction and the merge, so it sees the final id ->
        # name mapping for the round; an ENTITY_CORRECTION rename or a merge that
        # re-pointed an edge would otherwise leave the edge naming the old entity.
        self._sync_relationship_endpoint_names(merged_relationships, merged_entities)

        # A unit gained when it proposed an entity the graph did not have, or a
        # relationship that survived the merge (which re-points it in place)
        # under a key the graph did not have. Re-proposing known items is not a
        # gain; counting it would re-send the unit every round for nothing.
        kept_relationships = {id(r) for r in merged_relationships}
        new_items: list[Entity | Relationship] = [
            e for e in new_entities if entity_key(e.name) not in known_entity_keys
        ]
        new_items += [
            r
            for r in new_relationships
            if id(r) in kept_relationships
            and self._relationship_key(r) not in known_relationship_keys
        ]
        processed_ids = {unit.id for unit in text_units}
        gained_unit_ids = {
            unit_id
            for item in new_items
            for unit_id in item.text_unit_ids or []
            if unit_id in processed_ids
        }

        round_info = GleaningRound(
            round_number=round_num,
            units_gleaned=len(text_units),
            units_gained=len(gained_unit_ids),
            entities_before=entities_before,
            relationships_before=relationships_before,
            # Clamped: dropping an orphaned input edge can shrink the count.
            entities_added=max(0, len(merged_entities) - entities_before),
            relationships_added=max(
                0, len(merged_relationships) - relationships_before
            ),
            processing_time=time.time() - round_start_time,
        )
        return merged_entities, merged_relationships, round_info, gained_unit_ids

    def _perform_llm_refinement(
        self,
        text_units: list[TextUnit],
        current_entities: list[Entity],
        current_relationships: list[Relationship],
    ) -> tuple[list[Entity], list[Relationship]]:
        config_for_task = {
            "max_entities_per_prompt": self.max_entities_per_prompt,
            "max_relationships_per_prompt": self.max_relationships_per_prompt,
            "target_language": self.config.processing.translation.target_language.value,
        }

        task_with_args = partial(
            prepare_input_task,
            all_entities=current_entities,
            all_relationships=current_relationships,
            config=config_for_task,
        )

        executor_class = (
            ProcessPoolExecutor if self.use_process_pool else ContextThreadPoolExecutor
        )

        # Key results by their OWNING unit, not by completion order: as_completed
        # yields futures out of order, so appending to a list and zipping it
        # positionally against text_units would mismatch inputs to the wrong unit
        # (and crash under strict=True when a failed unit is skipped).
        unit_to_input: dict[str, Any] = {}
        with executor_class(max_workers=self.max_workers) as executor:
            future_to_unit = {
                executor.submit(task_with_args, unit): unit for unit in text_units
            }

            for future in tqdm(
                as_completed(future_to_unit),
                total=len(text_units),
                desc="Preparing Gleaning Inputs",
                disable=None if self.show_progress else True,
            ):
                unit = future_to_unit[future]
                try:
                    unit_to_input[unit.id] = future.result()
                except Exception as e:
                    logger.error(
                        "Error preparing gleaning input for unit '%s': %s", unit.id, e
                    )

        def prepare_inputs_for_chunk(
            chunk_items: list[TextUnit],
        ) -> list[dict[str, Any]]:
            return [
                unit_to_input[unit.id]
                for unit in chunk_items
                if unit.id in unit_to_input
            ]

        # Only the units whose input prep succeeded are actually processed, and
        # execute_with_fallback returns results 1:1 with THOSE inputs (not with
        # the full text_units list). Zipping the full list against the shorter
        # results with strict=True would raise and abort the whole stage — even
        # under ignore_errors, because the zip is outside the try below. Align
        # the zip to the prepared units so a single failed prep degrades to
        # skipping that unit, not crashing the stage.
        prepared_units = [u for u in text_units if u.id in unit_to_input]
        self._failed_unit_ids.extend(
            u.id for u in text_units if u.id not in unit_to_input
        )

        try:
            results = self.batch_processor.execute_with_fallback(
                items_to_process=prepared_units,
                prepare_inputs_func=prepare_inputs_for_chunk,
                batch_func=self.graph_refiner.batch,
                sequential_func=self.graph_refiner.invoke,
                task_name="Graph Refinement",
                run_config=self.config.processing.model_dump(),
                show_progress=self.show_progress,
            )
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.error("Error during graph refinement: %s", e)
            self._failed_unit_ids.extend(u.id for u in prepared_units)
            return [], []

        all_new_entities: list[Entity] = []
        all_new_relationships: list[Relationship] = []

        # This loop is serial, so a correction may mutate an entity or
        # relationship carried by the round in place without racing another chunk.
        for item, result_data in zip(prepared_units, results, strict=True):
            if result_data is BATCH_ITEM_FAILED:
                self._failed_unit_ids.append(item.id)
                continue
            new_entities, new_relationships = self._parse_refinement_output(
                result_data.get("refinement_plan", {}),
                item,
                current_entities,
                current_relationships,
            )
            all_new_entities.extend(new_entities)
            all_new_relationships.extend(new_relationships)

        if all_new_entities:
            all_entity_details = [f"'{entity.name}'" for entity in all_new_entities]
            logger.debug("All new entities: %s", all_entity_details)
        if all_new_relationships:
            all_relationship_details = [
                f"'{rel.source_name}' -> '{rel.target_name}' (type: '{rel.type}')"
                for rel in all_new_relationships
            ]
            logger.debug("All new relationships: %s", all_relationship_details)

        return all_new_entities, all_new_relationships

    def _parse_refinement_output(
        self,
        result_data: dict[str, Any] | list[Any],
        unit: TextUnit,
        existing_entities: list[Entity],
        existing_relationships: list[Relationship] | None = None,
    ) -> tuple[list[Entity], list[Relationship]]:
        new_entities: list[Entity] = []
        new_relationships: list[Relationship] = []

        if not result_data:
            logger.debug("No result data for text unit '%s'", unit.id)
            return new_entities, new_relationships

        try:
            plan = None
            if isinstance(result_data, dict):
                plan = result_data
            elif isinstance(result_data, list) and result_data:
                if isinstance(result_data[0], dict):
                    plan = result_data[0]

            if not isinstance(plan, dict):
                logger.warning(
                    "Could not extract a valid dictionary-based plan for unit '%s'. Received data type: %s, Preview: %s",
                    unit.id,
                    type(result_data),
                    str(result_data)[:250],
                )
                return new_entities, new_relationships

            issues_data = plan.get("identified_issues", {})
            if isinstance(issues_data, dict):
                issues = ensure_list(issues_data.get("issue", []))
            else:
                issues = ensure_list(issues_data)

            current_and_new_entities = list(existing_entities)

            for issue in issues:
                self._process_issue(
                    issue,
                    unit,
                    current_and_new_entities,
                    new_entities,
                    new_relationships,
                    existing_relationships or [],
                )

            if new_entities or new_relationships:
                logger.debug(
                    "Parsed refinement output for unit '%s': %s entities, %s relationships",
                    unit.id,
                    len(new_entities),
                    len(new_relationships),
                )
        except Exception as e:
            logger.warning(
                "Failed to parse refinement output for unit %s: %s. Input data: %s",
                unit.id,
                e,
                str(result_data)[:250],
            )

        return new_entities, new_relationships

    def _process_issue(
        self,
        issue: dict[str, Any],
        unit: TextUnit,
        current_and_new_entities: list[Entity],
        new_entities: list[Entity],
        new_relationships: list[Relationship],
        current_relationships: list[Relationship] | None = None,
    ) -> None:
        details = issue.get("details", {})
        issue_type = issue.get("issue_type", "").upper()

        # The refinement prompt requires a <text_evidence> exact quote per issue
        # (sibling of <details>). When grounding is enabled, reject additions
        # whose evidence is not actually in the chunk — the same hallucination
        # guard applied at first-pass extraction, now for gleaner-introduced
        # entities/relationships.
        if self.extraction_config.entity_grounding.enabled:
            text_evidence = issue.get("text_evidence")
            chunk_text = self.get_text_for_processing(unit)
            grounding = self.extraction_config.entity_grounding
            if not is_grounded(
                text_evidence if isinstance(text_evidence, str) else None,
                chunk_text,
                min_span_tokens=grounding.min_span_tokens,
                min_overlap_ratio=grounding.min_overlap_ratio,
            ):
                logger.info(
                    "Dropping ungrounded gleaner %s in chunk '%s' — text_evidence "
                    "not found (likely hallucinated)",
                    issue_type or "issue",
                    unit.short_id,
                )
                return

        if issue_type == "MISSING_ENTITY":
            entity = self.parse_entity_data(details, unit)
            if entity:
                # Strip the reserved grounding span (gleaner already verified it).
                (entity.attributes or {}).pop("_source_text", None)
                new_entities.append(entity)
                current_and_new_entities.append(entity)
        elif issue_type == "MISSING_RELATIONSHIP":
            entity_name_to_id = self.build_entity_key_index(current_and_new_entities)
            rel = self.parse_relationship_data(details, unit, entity_name_to_id)
            if rel:
                (rel.attributes or {}).pop("_source_text", None)
                new_relationships.append(rel)
        elif issue_type == "ENTITY_CORRECTION":
            self._apply_entity_correction(details, current_and_new_entities, unit)
        elif issue_type == "RELATIONSHIP_CORRECTION":
            self._apply_relationship_correction(
                details, current_relationships or [], unit
            )
        else:
            # Every type the refinement prompt asks for must have a branch here.
            # Falling through silently is how ENTITY_CORRECTION and
            # RELATIONSHIP_CORRECTION were discarded for every chunk of every run.
            logger.warning(
                "Ignoring gleaner issue of unhandled type '%s' in chunk '%s'",
                issue_type or "<missing>",
                unit.short_id,
            )

    @staticmethod
    def _clean_text(value: Any) -> str | None:
        return value.strip() or None if isinstance(value, str) else None

    @classmethod
    def _matches(cls, left: Any, right: Any) -> bool:
        """Compare two optional names by identity key (see ``entity_key``).

        Case-, whitespace-, quote- and trailing-punctuation-insensitive, so the
        LLM's "ACME Corp." matches the stored display name "Acme Corp".
        """
        left_clean, right_clean = cls._clean_text(left), cls._clean_text(right)
        if left_clean is None or right_clean is None:
            return False
        return entity_key(left_clean) == entity_key(right_clean)

    def _apply_entity_correction(
        self, details: dict[str, Any], entities: list[Entity], unit: TextUnit
    ) -> bool:
        """Apply an ENTITY_CORRECTION to the existing entity it names.

        The corrected entity is mutated in place, so the correction reaches the
        graph through the same list the round already carries. ``id`` is
        deliberately left alone: relationships reference it, and re-deriving it
        from a corrected name would orphan every edge touching this entity.

        Args:
            details: The issue's ``<details>`` payload.
            entities: Entities visible to this chunk (existing plus newly added).
            unit: The text unit being refined, for log attribution.

        Returns:
            True when at least one field changed.
        """
        name = self._clean_text(details.get("name"))
        entity = next((e for e in entities if self._matches(e.name, name)), None)
        if entity is None:
            logger.info(
                "Dropping ENTITY_CORRECTION in chunk '%s': no matching current entity",
                unit.short_id,
            )
            logger.debug("Unmatched ENTITY_CORRECTION name: '%s'", name)
            return False

        corrected: list[str] = []
        # Extracted names keep their display form (clean_display_name); a
        # corrected one gets the same cleaning so both spellings are comparable.
        corrected_name = clean_display_name(
            self._clean_text(details.get("corrected_name")) or ""
        )
        if corrected_name and corrected_name != entity.name:
            entity.name = corrected_name
            # The stored embedding described the old name.
            entity.name_embedding = None
            corrected.append("name")

        corrected_type = self._clean_text(details.get("corrected_type"))
        if corrected_type and corrected_type != entity.type:
            entity.type = corrected_type
            corrected.append("type")

        description = self._clean_text(details.get("description"))
        if description and description != entity.description:
            entity.description = description
            entity.description_embedding = None
            corrected.append("description")

        if not corrected:
            logger.debug(
                "ENTITY_CORRECTION for '%s' in chunk '%s' proposed no change",
                entity.name,
                unit.short_id,
            )
            return False

        entity.updated_at = datetime.now()
        logger.info(
            "Applied ENTITY_CORRECTION to entity '%s' in chunk '%s' (%s)",
            entity.short_id,
            unit.short_id,
            ", ".join(corrected),
        )
        return True

    def _apply_relationship_correction(
        self,
        details: dict[str, Any],
        relationships: list[Relationship],
        unit: TextUnit,
    ) -> bool:
        """Apply a RELATIONSHIP_CORRECTION to the existing edge it names.

        Scope matches what the prompt asks for: a wrong type, a reversed
        direction, or a wrong description. Re-pointing an edge at a different
        pair of entities is not a correction and is rejected — that is a
        MISSING_RELATIONSHIP plus a deletion, which this stage cannot express.

        Args:
            details: The issue's ``<details>`` payload.
            relationships: The relationships carried into this gleaning round.
            unit: The text unit being refined, for log attribution.

        Returns:
            True when at least one field changed.
        """
        source, target = details.get("source"), details.get("target")
        relationship = self._find_relationship(
            relationships, source, target, details.get("type")
        )
        if relationship is None:
            logger.info(
                "Dropping RELATIONSHIP_CORRECTION in chunk '%s': no single current "
                "relationship '%s' -> '%s'",
                unit.short_id,
                source,
                target,
            )
            return False

        corrected: list[str] = []
        corrected_type = self._clean_text(details.get("corrected_type"))
        if corrected_type and corrected_type != relationship.type:
            relationship.type = corrected_type
            corrected.append("type")

        # A reversal is the corrected endpoints naming the current pair the other
        # way round. Both ids and names move together, or the edge would point at
        # one entity while claiming to name another.
        if self._matches(
            details.get("corrected_source"), relationship.target_name
        ) and self._matches(details.get("corrected_target"), relationship.source_name):
            relationship.source_id, relationship.target_id = (
                relationship.target_id,
                relationship.source_id,
            )
            relationship.source_name, relationship.target_name = (
                relationship.target_name,
                relationship.source_name,
            )
            corrected.append("direction")

        description = self._clean_text(details.get("description"))
        if description and description != relationship.description:
            relationship.description = description
            relationship.description_embedding = None
            corrected.append("description")

        if not corrected:
            logger.debug(
                "RELATIONSHIP_CORRECTION for '%s' -> '%s' in chunk '%s' proposed no "
                "change",
                relationship.source_name,
                relationship.target_name,
                unit.short_id,
            )
            return False

        relationship.updated_at = datetime.now()
        logger.info(
            "Applied RELATIONSHIP_CORRECTION to '%s' -> '%s' in chunk '%s' (%s)",
            relationship.source_name,
            relationship.target_name,
            unit.short_id,
            ", ".join(corrected),
        )
        return True

    @classmethod
    def _find_relationship(
        cls,
        relationships: list[Relationship],
        source: Any,
        target: Any,
        current_type: Any = None,
    ) -> Relationship | None:
        """Resolve the edge a correction names, or None when it is ambiguous."""
        candidates = [
            rel
            for rel in relationships
            if cls._matches(rel.source_name, source)
            and cls._matches(rel.target_name, target)
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # A pair can carry several edges; the stated current type disambiguates.
        typed = [rel for rel in candidates if cls._matches(rel.type, current_type)]
        return typed[0] if len(typed) == 1 else None

    @staticmethod
    def _merge_duplicate_entities(
        entities: list[Entity],
    ) -> tuple[list[Entity], dict[str, str]]:
        entities_map: dict[str, Entity] = {}
        id_remap: dict[str, str] = {}
        type_counts: dict[str, dict[str, int]] = {}
        merged_names = []

        for entity in entities:
            # Same identity key as the entity id, so display variants
            # ("Acme Corp" / "ACME Corp.") merge while "C++" / "C#" do not.
            key = entity_key(entity.name)
            if key not in entities_map:
                entities_map[key] = entity
                type_counts[key] = {}
                entity_type = entity.type.lower() if entity.type else ""
                type_counts[key][entity_type] = 1
            else:
                master_entity = entities_map[key]
                id_remap[entity.id] = master_entity.id
                merged_names.append(entity.name)
                entity_type = entity.type.lower() if entity.type else ""
                type_counts[key][entity_type] = type_counts[key].get(entity_type, 0) + 1

                if entity.description and entity.description.strip():
                    if master_entity.description and master_entity.description.strip():
                        master_entity.description = (
                            f"{master_entity.description}; {entity.description}"
                        )
                    else:
                        master_entity.description = entity.description

                master_entity.text_unit_ids = list(
                    set(
                        (master_entity.text_unit_ids or [])
                        + (entity.text_unit_ids or [])
                    )
                )

        for key, entity in entities_map.items():
            if type_counts[key]:
                most_frequent_type = max(
                    type_counts[key].keys(), key=lambda t: type_counts[key][t]
                )
                entity.type = most_frequent_type

        unique_entities = list(entities_map.values())
        duplicates_merged = len(entities) - len(unique_entities)

        if duplicates_merged > 0:
            logger.debug(
                "Merged %s duplicate entities (%s -> %s)",
                duplicates_merged,
                len(entities),
                len(unique_entities),
            )
            for name in merged_names:
                logger.debug("Merged duplicate entity: '%s'", name)

        return unique_entities, id_remap

    @staticmethod
    def _update_relationships_after_merge(
        relationships: list[Relationship],
        unique_entity_ids: set[str],
        id_remap: dict[str, str],
    ) -> list[Relationship]:
        relationships_map: dict[tuple, Relationship] = {}
        dropped = 0

        for rel in relationships:
            source_id = id_remap.get(rel.source_id, rel.source_id)
            target_id = id_remap.get(rel.target_id, rel.target_id)

            if (
                source_id in unique_entity_ids
                and target_id in unique_entity_ids
                and source_id != target_id
            ):
                rel.source_id = source_id
                rel.target_id = target_id
                key = GraphGleaner._relationship_key(rel)

                if key not in relationships_map:
                    relationships_map[key] = rel
                else:
                    existing_rel = relationships_map[key]
                    if rel.description and rel.description.strip():
                        if (
                            existing_rel.description
                            and existing_rel.description.strip()
                        ):
                            existing_rel.description = (
                                f"{existing_rel.description}; {rel.description}"
                            )
                        else:
                            existing_rel.description = rel.description

                    # Use `is not None` (not `or`): a legitimate weight of 0.0
                    # must not be silently promoted to the 1.0 default.
                    existing_w = (
                        existing_rel.weight if existing_rel.weight is not None else 1.0
                    )
                    delta_w = rel.weight if rel.weight is not None else 1.0
                    existing_rel.weight = existing_w + delta_w
            else:
                # Endpoint merged away / not resolved, or a self-loop after
                # remap: the relationship cannot be kept. Count it so dropped
                # edges are visible rather than silently vanishing.
                dropped += 1

        final_relationships = list(relationships_map.values())
        duplicates_merged = len(relationships) - len(final_relationships) - dropped

        if duplicates_merged > 0 or dropped > 0:
            logger.debug(
                "Relationship post-merge: %s in -> %s out (%s merged, %s dropped "
                "as orphaned/self-loop)",
                len(relationships),
                len(final_relationships),
                duplicates_merged,
                dropped,
            )

        return final_relationships

    @staticmethod
    def _sync_relationship_endpoint_names(
        relationships: list[Relationship], entities: list[Entity]
    ) -> None:
        """Rewrite each edge's endpoint names from the final entity id -> name map.

        ``source_name`` / ``target_name`` are denormalised copies of the entity
        names, and the refinement prompt and the indexers read those copies
        rather than resolving the id. An ENTITY_CORRECTION renames an entity in
        place and keeps its id, and a merge re-points an edge at the surviving
        entity's id; neither touches the copies, so once both have run the names
        are re-derived from the ids the edge references.
        """
        name_by_id = {entity.id: entity.name for entity in entities}
        resynced = 0
        for rel in relationships:
            source_name = name_by_id.get(rel.source_id, rel.source_name)
            target_name = name_by_id.get(rel.target_id, rel.target_name)
            if (source_name, target_name) != (rel.source_name, rel.target_name):
                rel.source_name, rel.target_name = source_name, target_name
                resynced += 1

        if resynced > 0:
            logger.debug(
                "Resynced endpoint names on %s relationships after entity "
                "corrections/merges",
                resynced,
            )

    @staticmethod
    def _log_completion_summary(stats: GleaningStats, units_still_gaining: int) -> None:
        logger.info(
            "Graph gleaning completed: %s rounds, %s refinement calls, "
            "%s entities added, %s relationships added in %.2fs",
            stats.total_rounds,
            stats.total_refinement_calls,
            stats.total_entities_added,
            stats.total_relationships_added,
            stats.total_processing_time,
        )
        if units_still_gaining:
            logger.info(
                "Stopped at max_rounds with %s text units still gaining items",
                units_still_gaining,
            )
        if stats.num_failed_units > 0:
            logger.warning(
                "Gleaning failed for %s text-unit refinements", stats.num_failed_units
            )
