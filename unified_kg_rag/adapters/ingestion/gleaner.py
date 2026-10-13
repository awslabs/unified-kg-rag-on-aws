# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

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
    union_document_filters,
)
from unified_kg_rag.domain.ingestion.descriptions import merge_descriptions
from unified_kg_rag.domain.ingestion.entity_grounding import (
    is_grounded,
    is_name_grounded,
)
from unified_kg_rag.domain.ingestion.relationship_endpoints import (
    cite_relationship_endpoints,
    sync_relationship_endpoint_names,
)
from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
    text_unit_weights,
)
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    ModelPurpose,
    RejectedEntity,
    Relationship,
    TextUnit,
)
from unified_kg_rag.domain.prompts import GraphRefinementPrompt
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import (
    clean_display_name,
    ensure_list,
    entity_key,
)
from unified_kg_rag.shared.utils.langchain import BATCH_ITEM_FAILED, BatchProcessor

logger = get_logger(__name__)


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


def prepare_input(
    unit: TextUnit,
    unit_entities: list[Entity],
    unit_relationships: list[Relationship],
    config: dict[str, Any],
) -> dict[str, Any]:
    """The refinement prompt's input for ``unit`` and the graph it cites."""
    target_language = config.get("target_language", "en")
    if unit.translated_texts:
        unit_text = unit.translated_texts.get(target_language, unit.text or "")
    else:
        unit_text = unit.text or ""

    return {
        "text": unit_text,
        "entities": format_entities_with_limit_task(
            unit_entities, config["max_entities_per_prompt"]
        ),
        "relationships": format_relationships_with_limit_task(
            unit_relationships, config["max_relationships_per_prompt"]
        ),
        "target_language": target_language,
    }


def _name_key(value: Any) -> str | None:
    """``entity_key`` of a non-blank string, else None (matches nothing)."""
    return entity_key(value) if isinstance(value, str) and value.strip() else None


class GleaningIndex:
    """The lookups one gleaning round needs, built once from its graph.

    Parsing a unit's answer resolves names to entities and edges; scanning
    the whole graph for each would make a round quadratic in its size. The
    index follows the changes parsing makes: the entities an answer adds, a
    rename (the entity moves to its new name key) and a reversal (the edge
    moves to its new endpoint pair).
    """

    def __init__(
        self, entities: list[Entity], relationships: list[Relationship]
    ) -> None:
        self._entities_by_unit = group_by_text_unit(entities)
        self._relationships_by_unit = group_by_text_unit(relationships)
        self._entities_by_key: dict[str, list[Entity]] = {}
        # entity_key -> id of the first entity with that key: the endpoint
        # index parse_relationship_data reads.
        self.entity_ids: dict[str, str] = {}
        for entity in entities:
            self._add_key(entity)
        self._relationships_by_pair: dict[tuple[str, str], list[Relationship]] = {}
        for rel in relationships:
            self._relationships_by_pair.setdefault(self._pair(rel), []).append(rel)

    @staticmethod
    def _pair(rel: Relationship) -> tuple[str, str]:
        return entity_key(rel.source_name or ""), entity_key(rel.target_name or "")

    def _add_key(self, entity: Entity) -> None:
        key = entity_key(entity.name)
        self._entities_by_key.setdefault(key, []).append(entity)
        self.entity_ids.setdefault(key, entity.id)

    def entities_for(self, unit_id: str) -> list[Entity]:
        """Entities citing the unit (its lineage), in graph order."""
        return self._entities_by_unit.get(unit_id, [])

    def relationships_for(self, unit_id: str) -> list[Relationship]:
        """Relationships citing the unit, in graph order."""
        return self._relationships_by_unit.get(unit_id, [])

    def entity_named(self, name: Any) -> Entity | None:
        """The first entity whose name has the same ``entity_key``."""
        key = _name_key(name)
        matches = self._entities_by_key.get(key, []) if key else []
        return matches[0] if matches else None

    def relationships_between(self, source: Any, target: Any) -> list[Relationship]:
        """Edges whose endpoint names have these ``entity_key`` values."""
        source_key, target_key = _name_key(source), _name_key(target)
        if source_key is None or target_key is None:
            return []
        return self._relationships_by_pair.get((source_key, target_key), [])

    def add_entity(self, entity: Entity) -> None:
        self._add_key(entity)
        for unit_id in entity.text_unit_ids or []:
            self._entities_by_unit.setdefault(unit_id, []).append(entity)

    def rename_entity(self, entity: Entity, old_name: str) -> None:
        old_key = entity_key(old_name)
        remaining = [
            e for e in self._entities_by_key.get(old_key, []) if e is not entity
        ]
        self._entities_by_key[old_key] = remaining
        if remaining:
            self.entity_ids[old_key] = remaining[0].id
        else:
            self.entity_ids.pop(old_key, None)
        self._add_key(entity)

    def move_relationship(self, rel: Relationship, old_pair: tuple[str, str]) -> None:
        self._relationships_by_pair[old_pair] = [
            r for r in self._relationships_by_pair.get(old_pair, []) if r is not rel
        ]
        self._relationships_by_pair.setdefault(self._pair(rel), []).append(rel)


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
    # Gleaned relationships dropped because an endpoint is an entity rejected
    # for the relationship's unit: dropped by extraction's grounding guard or
    # confidence threshold, or proposed by gleaning with evidence not found in
    # the unit or a confidence below the threshold.
    relationships_dropped_rejected_endpoint: int = 0
    # Gleaned entities dropped because their confidence is below
    # entity_confidence_threshold, or the threshold removed them from their
    # unit at extraction.
    entities_dropped_low_confidence: int = 0
    # Gleaned relationships dropped (entity grounding on) because an endpoint
    # is an entity the unit does not list and whose name it does not contain.
    relationships_dropped_ungrounded_endpoint_name: int = 0
    rounds: list[GleaningRound] = Field(default_factory=list)

    @property
    def total_refinement_calls(self) -> int:
        return sum(r.units_gleaned for r in self.rounds)


class GraphGleaner(BaseProcessor):
    def __init__(
        self,
        config: Config,
        show_progress: bool = True,
        *,
        providers: Providers | None = None,
    ) -> None:
        super().__init__(config)
        self.providers = Providers.resolve(config, providers)
        self.gleaning_config = self.config.processing.gleaning
        self.ignore_errors = self.config.processing.ignore_errors
        self.show_progress = show_progress
        self._failed_unit_ids: list[str] = []
        # Entity keys rejected per text unit (by extraction, or by gleaning's
        # grounding check in an earlier round); see
        # _drop_unsupported_relationships.
        self._rejected_keys: dict[str, set[str]] = {}
        # Entity keys below the confidence threshold per text unit; unlike
        # _rejected_keys, a later grounded copy does not lift them.
        self._low_confidence_keys: dict[str, set[str]] = {}
        self._dropped_rejected_endpoint = 0
        self._dropped_low_confidence_entities = 0
        self._dropped_ungrounded_endpoint_name = 0

        self.factory = self.providers.llm_factory
        self.batch_processor = BatchProcessor.from_processing(
            self.config.processing,
            is_transient_error=transient_error_predicate(self.factory),
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
            min_output_tokens=GraphRefinementPrompt.output_floor(self.config),
            output_tags=["refinement_plan"],
        )
        self.graph_refiner = setup_chain(
            model_purpose=ModelPurpose.INGESTION,
            factory=self.factory,
            model_id=self.gleaning_config.graph_refinement_model_id,
            prompt_class=GraphRefinementPrompt,
            min_output_tokens=GraphRefinementPrompt.output_floor(self.config),
            parser=robust_xml_output_parser,
            custom_prompts=self.config.custom_prompts,
        )

    def glean_graph(
        self,
        text_units: list[TextUnit],
        initial_entities: list[Entity],
        initial_relationships: list[Relationship],
        *,
        rejected_entities: Sequence[RejectedEntity] = (),
    ) -> tuple[list[Entity], list[Relationship], GleaningStats]:
        """Glean each text unit for at most ``max_rounds`` rounds.

        Round 1 sends every unit. Each later round re-sends only the units
        whose previous answer added a new entity or relationship. A unit stops
        once its answer adds nothing new: an empty ``<identified_issues>`` is
        the model saying nothing more is missing (the answer MS GraphRAG gets
        from its separate Y/N loop prompt), and items that merge into existing
        ones or fail grounding are not new.

        ``rejected_entities`` are the entities extraction rejected per unit; a
        relationship gleaned from that unit naming one is dropped rather than
        bringing the entity back as a stub, and an entity the confidence
        threshold removed is not re-added from that unit at all.
        """
        start_time = time.time()
        current_entities = initial_entities.copy()
        current_relationships = initial_relationships.copy()
        stats = GleaningStats()
        self._failed_unit_ids = []
        self._rejected_keys = {}
        self._low_confidence_keys = {}
        for rejected in rejected_entities:
            keys = (
                self._low_confidence_keys
                if rejected.reason == "low_confidence"
                else self._rejected_keys
            )
            keys.setdefault(rejected.text_unit_id, set()).add(rejected.entity_key)
        self._dropped_rejected_endpoint = 0
        self._dropped_low_confidence_entities = 0
        self._dropped_ungrounded_endpoint_name = 0
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
        stats.relationships_dropped_rejected_endpoint = self._dropped_rejected_endpoint
        stats.entities_dropped_low_confidence = self._dropped_low_confidence_entities
        stats.relationships_dropped_ungrounded_endpoint_name = (
            self._dropped_ungrounded_endpoint_name
        )
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

        merged_entities, merged_relationships = self._merge_round(
            current_entities + new_entities,
            current_relationships + new_relationships,
        )

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

    @classmethod
    def _merge_round(
        cls,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> tuple[list[Entity], list[Relationship]]:
        """Merge a round's additions into the graph it was gleaned from.

        Endpoints first cite their relationships' text units, as after
        first-pass extraction (see :func:`cite_relationship_endpoints`): an
        added edge naming an entity another chunk extracted makes that entity
        cite this chunk, and one naming an entity no chunk listed adds it as a
        stub (with entity grounding on, only one whose name the chunk contains
        gets here: see :meth:`_drop_unsupported_relationships`). Otherwise the endpoint would not cite the chunk while the
        document lineage still attributes it to the chunk's document, so
        removing the entity's other documents would leave it citing nothing,
        and the edge would depend on which documents shared the batch.
        """
        entities, stub_count = cite_relationship_endpoints(entities, relationships)
        if stub_count:
            logger.debug(
                "Materialized %s entities referenced only by gleaned relationships",
                stub_count,
            )
        merged_entities, entity_id_map = cls._merge_duplicate_entities(entities)
        merged_relationships = cls._update_relationships_after_merge(
            relationships,
            {e.id for e in merged_entities},
            entity_id_map,
        )
        # Runs after every correction and the merge, so it sees the final id ->
        # name mapping for the round; an ENTITY_CORRECTION rename or a merge that
        # re-pointed an edge would otherwise leave the edge naming the old entity.
        resynced = sync_relationship_endpoint_names(
            merged_relationships, merged_entities
        )
        if resynced:
            logger.debug(
                "Resynced endpoint names on %s relationships after entity "
                "corrections/merges",
                resynced,
            )
        return merged_entities, merged_relationships

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
        # Built once per round: each unit's prompt lists the entities and
        # edges citing it, and parsing its answer resolves names against the
        # whole graph. Preparing inputs is a dict lookup and a format per
        # unit, so it runs here rather than shipping the graph to workers.
        index = GleaningIndex(current_entities, current_relationships)
        unit_to_input: dict[str, Any] = {}
        for unit in text_units:
            try:
                unit_to_input[unit.id] = prepare_input(
                    unit,
                    index.entities_for(unit.id),
                    index.relationships_for(unit.id),
                    config_for_task,
                )
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
                sequential_func=self.graph_refiner.invoke,
                task_name="Graph Refinement",
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
                result_data.get("refinement_plan", {}), item, index
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
        index: GleaningIndex,
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
                    "Could not extract a valid dictionary-based plan for unit "
                    "'%s'. Received data type: %s",
                    unit.id,
                    type(result_data).__name__,
                )
                logger.debug("Refinement output preview: %s", str(result_data)[:250])
                return new_entities, new_relationships

            issues_data = plan.get("identified_issues", {})
            if isinstance(issues_data, dict):
                issues = ensure_list(issues_data.get("issue", []))
            else:
                issues = ensure_list(issues_data)

            rejected_keys: set[str] = set()
            penalized: list[Relationship] = []

            for issue in issues:
                self._process_issue(
                    issue,
                    unit,
                    index,
                    new_entities,
                    new_relationships,
                    rejected_keys=rejected_keys,
                    penalized_relationships=penalized,
                )

            new_relationships += self._keep_penalized_relationships(
                penalized, unit, index
            )
            new_relationships = self._drop_unsupported_relationships(
                new_relationships, unit, index.entities_for(unit.id), rejected_keys
            )
            if rejected_keys:
                self._rejected_keys.setdefault(unit.id, set()).update(rejected_keys)

            if new_entities or new_relationships:
                logger.debug(
                    "Parsed refinement output for unit '%s': %s entities, %s relationships",
                    unit.id,
                    len(new_entities),
                    len(new_relationships),
                )
        except Exception as e:
            logger.warning(
                "Failed to parse refinement output for unit %s: %s",
                unit.id,
                type(e).__name__,
            )
            logger.debug(
                "Refinement parse error: %s. Output preview: %s",
                e,
                str(result_data)[:250],
            )

        return new_entities, new_relationships

    def _process_issue(
        self,
        issue: dict[str, Any],
        unit: TextUnit,
        index: GleaningIndex,
        new_entities: list[Entity],
        new_relationships: list[Relationship],
        *,
        rejected_keys: set[str] | None = None,
        penalized_relationships: list[Relationship] | None = None,
    ) -> None:
        """Apply one refinement issue to the round's additions.

        An issue whose ``text_evidence`` the grounding check does not find in
        the chunk follows ``entity_grounding.action`` as at extraction. Under
        ``drop`` (and for a correction under either action) it is ignored,
        and the name key of a rejected MISSING_ENTITY is added to
        ``rejected_keys``, so a relationship naming it can be dropped (see
        :meth:`_drop_unsupported_relationships`). Under ``penalize`` a
        MISSING_ENTITY is added with its confidence times ``penalty_factor``,
        and a MISSING_RELATIONSHIP goes to ``penalized_relationships`` with
        its weight times ``penalty_factor`` (see
        :meth:`_keep_penalized_relationships`). A MISSING_ENTITY below
        ``entity_confidence_threshold``, or one the threshold removed from
        this unit at extraction, is dropped as extraction's filter would drop
        it, and its key is held for the unit like a filtered entity's.
        """
        details = issue.get("details", {})
        issue_type = issue.get("issue_type", "").upper()
        grounding = self.extraction_config.entity_grounding
        penalize = False

        # The refinement prompt requires a <text_evidence> exact quote per issue
        # (sibling of <details>). When grounding is enabled, reject additions
        # whose evidence is not actually in the chunk — the same hallucination
        # guard applied at first-pass extraction, now for gleaner-introduced
        # entities/relationships.
        if grounding.enabled:
            text_evidence = issue.get("text_evidence")
            chunk_text = self.get_text_for_processing(unit)
            grounded = is_grounded(
                text_evidence if isinstance(text_evidence, str) else None,
                chunk_text,
                min_span_tokens=grounding.min_span_tokens,
                min_overlap_ratio=grounding.min_overlap_ratio,
            )
            penalize = (
                not grounded
                and grounding.action == "penalize"
                and issue_type in ("MISSING_ENTITY", "MISSING_RELATIONSHIP")
            )
            if not grounded and not penalize:
                logger.info(
                    "Dropping ungrounded gleaner %s in chunk '%s' — text_evidence "
                    "not found (likely hallucinated)",
                    issue_type or "issue",
                    unit.short_id,
                )
                if (
                    issue_type == "MISSING_ENTITY"
                    and rejected_keys is not None
                    and isinstance(details, dict)
                ):
                    name = clean_display_name(coerce_llm_text(details.get("name")))
                    if name:
                        rejected_keys.add(entity_key(name))
                return

        if issue_type == "MISSING_ENTITY":
            entity = self.parse_entity_data(details, unit)
            if entity and penalize:
                base = entity.confidence if entity.confidence is not None else 1.0
                entity.confidence = base * grounding.penalty_factor
            if entity and self._below_confidence_threshold(entity, unit):
                return
            if entity:
                # Strip the reserved grounding span (gleaner already verified it).
                (entity.attributes or {}).pop("_source_text", None)
                new_entities.append(entity)
                index.add_entity(entity)
        elif issue_type == "MISSING_RELATIONSHIP":
            rel = self.parse_relationship_data(details, unit, index.entity_ids)
            if rel:
                (rel.attributes or {}).pop("_source_text", None)
                if penalize and penalized_relationships is not None:
                    rel.weight = (rel.weight or 1.0) * grounding.penalty_factor
                    penalized_relationships.append(rel)
                else:
                    new_relationships.append(rel)
        elif issue_type == "ENTITY_CORRECTION":
            self._apply_entity_correction(details, index, unit)
        elif issue_type == "RELATIONSHIP_CORRECTION":
            self._apply_relationship_correction(details, index, unit)
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
    def _keep_penalized_relationships(
        relationships: list[Relationship], unit: TextUnit, index: GleaningIndex
    ) -> list[Relationship]:
        """The penalized (ungrounded) relationships whose endpoints cite ``unit``.

        Extraction's ``penalize`` rule: an ungrounded relationship is kept
        only when the chunk lists both endpoints, since keeping one would make
        an endpoint the chunk did not list cite it (see
        ``GraphExtractor._apply_relationship_grounding``). Checked after every
        issue of the response, so the order of the issues does not matter.
        """
        if not relationships:
            return []
        listed = {entity.id for entity in index.entities_for(unit.id)}
        kept: list[Relationship] = []
        for rel in relationships:
            if {rel.source_id, rel.target_id} <= listed:
                kept.append(rel)
                continue
            logger.info(
                "Dropping ungrounded gleaned relationship '%s' — it names an "
                "entity chunk '%s' does not list",
                rel.short_id,
                unit.short_id,
            )
        return kept

    def _drop_unsupported_relationships(
        self,
        relationships: list[Relationship],
        unit: TextUnit,
        entities: list[Entity],
        rejected_keys: set[str],
    ) -> list[Relationship]:
        """Drop gleaned relationships whose endpoint the unit does not support.

        ``entities`` are the entities citing ``unit``.

        Every kept relationship makes its endpoints cite its unit, adding a
        stub for one the batch lacks (:func:`cite_relationship_endpoints`), so
        a relationship naming an entity rejected for this unit would bring it
        back. Rejected means dropped as ungrounded by extraction for this
        unit, or proposed by gleaning (this or an earlier round) with
        evidence not found in it; the grounding check on the relationship
        covers only its quote, which need not name its endpoints. A rejected
        entity is supported again once an entity of that name cites the unit
        (the unit listed it with grounded evidence). Extraction drops the same
        relationships (``_drop_relationships_to_dropped_entities``).

        An entity the confidence threshold removed for this unit is rejected
        whatever else cites the unit, so a stub never resurrects it. A stub
        has no score of its own: it gets the default confidence (1.0), as an
        entity the model lists without one does; it only exists for an entity
        not rejected for its unit, so it never stands in for a filtered one.

        With entity grounding on, an endpoint no entity citing the unit names
        must also be named in the unit's text, whether or not another unit
        lists it: requiring that only of a stub would make the outcome depend
        on which units share the batch, which an incremental run cannot
        reproduce. With grounding off nothing is rejected and every
        relationship is kept.
        """
        if not relationships:
            return relationships
        grounding = self.extraction_config.entity_grounding
        cited = {entity_key(e.name) for e in entities}
        blocked = (
            (self._rejected_keys.get(unit.id, set()) | rejected_keys) - cited
        ) | self._low_confidence_keys.get(unit.id, set())
        if not blocked and not grounding.enabled:
            return relationships
        chunk_text = self.get_text_for_processing(unit) if grounding.enabled else ""

        kept: list[Relationship] = []
        for rel in relationships:
            names = (rel.source_name or "", rel.target_name or "")
            keys = tuple(entity_key(name) for name in names)
            if any(key in blocked for key in keys):
                self._dropped_rejected_endpoint += 1
                logger.info(
                    "Dropping gleaned relationship '%s' — an endpoint was "
                    "rejected (ungrounded or low confidence) in chunk '%s'",
                    rel.short_id,
                    unit.short_id,
                )
                continue
            if grounding.enabled and any(
                key not in cited and not is_name_grounded(name, chunk_text)
                for name, key in zip(names, keys, strict=True)
            ):
                self._dropped_ungrounded_endpoint_name += 1
                logger.info(
                    "Dropping gleaned relationship '%s' — chunk '%s' neither "
                    "lists nor names an endpoint",
                    rel.short_id,
                    unit.short_id,
                )
                continue
            kept.append(rel)
        return kept

    def _below_confidence_threshold(self, entity: Entity, unit: TextUnit) -> bool:
        """Whether a gleaned entity fails the extraction confidence threshold.

        True when its confidence is below ``entity_confidence_threshold`` or
        the threshold removed an entity of that name from ``unit`` at
        extraction (the gleaner's answer carries no score, so it cannot
        overturn the extraction's). Records the key for the unit, so a
        relationship naming the entity is dropped too.
        """
        key = entity_key(entity.name)
        held = self._low_confidence_keys.setdefault(unit.id, set())
        threshold = self.extraction_config.entity_confidence_threshold
        confidence = entity.confidence if entity.confidence is not None else 1.0
        if key not in held and (threshold <= 0.0 or confidence >= threshold):
            return False
        held.add(key)
        self._dropped_low_confidence_entities += 1
        logger.info(
            "Dropping gleaned entity '%s' in chunk '%s' — below the confidence "
            "threshold %s",
            entity.short_id,
            unit.short_id,
            threshold,
        )
        return True

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
        self, details: dict[str, Any], index: GleaningIndex, unit: TextUnit
    ) -> bool:
        """Apply an ENTITY_CORRECTION to the existing entity it names.

        The corrected entity is mutated in place, so the correction reaches the
        graph through the same list the round already carries. ``id`` is
        deliberately left alone: relationships reference it, and re-deriving it
        from a corrected name would orphan every edge touching this entity.

        The prompt lists entity names only, so the model never sees the
        current description: a proposed one is merged into it
        (``merge_descriptions``) rather than replacing it.

        Args:
            details: The issue's ``<details>`` payload.
            index: The round's graph (existing plus newly added entities).
            unit: The text unit being refined, for log attribution.

        Returns:
            True when at least one field changed.
        """
        name = self._clean_text(details.get("name"))
        entity = index.entity_named(name)
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
            old_name, entity.name = entity.name, corrected_name
            index.rename_entity(entity, old_name)
            # The stored embedding described the old name.
            entity.name_embedding = None
            corrected.append("name")

        corrected_type = self._clean_text(details.get("corrected_type"))
        if corrected_type and corrected_type != entity.type:
            entity.type = corrected_type
            corrected.append("type")

        description = merge_descriptions(
            (entity.description, self._clean_text(details.get("description")))
        )
        if description != entity.description:
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
        index: GleaningIndex,
        unit: TextUnit,
    ) -> bool:
        """Apply a RELATIONSHIP_CORRECTION to the existing edge it names.

        Scope matches what the prompt asks for: a wrong type, a reversed
        direction, or a wrong description. Re-pointing an edge at a different
        pair of entities is not a correction and is rejected — that is a
        MISSING_RELATIONSHIP plus a deletion, which this stage cannot express.
        A proposed description is merged into the current one, which the
        prompt does not show.

        Args:
            details: The issue's ``<details>`` payload.
            index: The round's graph (the relationships carried into it).
            unit: The text unit being refined, for log attribution.

        Returns:
            True when at least one field changed.
        """
        source, target = details.get("source"), details.get("target")
        relationship = self._find_relationship(
            index.relationships_between(source, target), details.get("type")
        )
        if relationship is None:
            logger.info(
                "Dropping RELATIONSHIP_CORRECTION in chunk '%s': no single current "
                "relationship matches its endpoints",
                unit.short_id,
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
            old_pair = GleaningIndex._pair(relationship)
            relationship.source_id, relationship.target_id = (
                relationship.target_id,
                relationship.source_id,
            )
            relationship.source_name, relationship.target_name = (
                relationship.target_name,
                relationship.source_name,
            )
            index.move_relationship(relationship, old_pair)
            corrected.append("direction")

        description = merge_descriptions(
            (relationship.description, self._clean_text(details.get("description")))
        )
        if description != relationship.description:
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
            "Applied RELATIONSHIP_CORRECTION to '%s' in chunk '%s' (%s)",
            relationship.short_id,
            unit.short_id,
            ", ".join(corrected),
        )
        return True

    @classmethod
    def _find_relationship(
        cls, candidates: list[Relationship], current_type: Any = None
    ) -> Relationship | None:
        """Pick the edge a correction names among the pair's edges.

        None when the pair has no edge, or several and the stated current type
        does not single one out.
        """
        if len(candidates) <= 1:
            return candidates[0] if candidates else None
        typed = [rel for rel in candidates if cls._matches(rel.type, current_type)]
        return typed[0] if len(typed) == 1 else None

    @staticmethod
    def _merge_duplicate_entities(
        entities: list[Entity],
    ) -> tuple[list[Entity], dict[str, str]]:
        entities_map: dict[str, Entity] = {}
        # Id -> the entities_map key of the entity holding it. An
        # ENTITY_CORRECTION renames an entity and keeps its id, so a later
        # proposal of the old name carries that id under another name key: it
        # merges into the renamed entity instead of becoming a second entity
        # with the same id.
        key_by_id: dict[str, str] = {}
        id_remap: dict[str, str] = {}
        type_counts: dict[str, dict[str, int]] = {}
        merged_names = []

        for entity in entities:
            # Same identity key as the entity id, so display variants
            # ("Acme Corp" / "ACME Corp.") merge while "C++" / "C#" do not.
            key = key_by_id.get(entity.id) or entity_key(entity.name)
            key_by_id.setdefault(entity.id, key)
            if key not in entities_map:
                entities_map[key] = entity
                type_counts[key] = {}
                entity_type = entity.type.lower() if entity.type else ""
                type_counts[key][entity_type] = 1
            else:
                master_entity = entities_map[key]
                if entity.id != master_entity.id:
                    id_remap[entity.id] = master_entity.id
                merged_names.append(entity.name)
                entity_type = entity.type.lower() if entity.type else ""
                type_counts[key][entity_type] = type_counts[key].get(entity_type, 0) + 1

                master_entity.description = merge_descriptions(
                    (master_entity.description, entity.description)
                )
                master_entity.text_unit_ids = list(
                    dict.fromkeys(
                        (master_entity.text_unit_ids or [])
                        + (entity.text_unit_ids or [])
                    )
                )
                master_entity.attributes = union_document_filters(
                    master_entity.attributes, entity.attributes
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
        # Id -> key of the kept edge holding it: a RELATIONSHIP_CORRECTION
        # retypes or reverses an edge and keeps its id, so a re-proposal of
        # the old edge merges into it rather than adding a second edge with
        # the same id.
        key_by_id: dict[str, tuple] = {}
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
                key = key_by_id.get(rel.id) or GraphGleaner._relationship_key(rel)
                key_by_id.setdefault(rel.id, key)

                if key not in relationships_map:
                    relationships_map[key] = rel
                else:
                    existing_rel = relationships_map[key]
                    existing_rel.description = merge_descriptions(
                        (existing_rel.description, rel.description)
                    )
                    existing_rel.attributes = union_document_filters(
                        existing_rel.attributes, rel.attributes
                    )
                    # A gleaned edge adds the strength of each text unit it
                    # newly cites. One the edge already cites keeps its stored
                    # strength: the model re-stating an edge it was shown is
                    # not a second instance of it.
                    stored = text_unit_weights(existing_rel)
                    weights = stored | {
                        unit_id: weight
                        for unit_id, weight in text_unit_weights(rel).items()
                        if unit_id not in stored
                    }
                    if weights:
                        apply_text_unit_weights(existing_rel, weights)
                    else:
                        # No lineage: plain sum. `is not None` keeps a real 0.0.
                        existing_rel.weight = (
                            existing_rel.weight
                            if existing_rel.weight is not None
                            else 1.0
                        ) + (rel.weight if rel.weight is not None else 1.0)
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
