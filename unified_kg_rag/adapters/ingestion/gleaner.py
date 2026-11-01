# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from unified_kg_rag.adapters.llm.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from unified_kg_rag.adapters.llm.langchain import BATCH_ITEM_FAILED, BatchProcessor
from unified_kg_rag.adapters.llm.retry import (
    item_error_predicate,
    item_failed_indices,
    transient_error_predicate,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.ingestion.base_processor import (
    BaseProcessor,
    coerce_llm_text,
    format_entities_with_limit_task,
    union_document_filters,
)
from unified_kg_rag.domain.ingestion.contributions import unit_entities
from unified_kg_rag.domain.ingestion.descriptions import merge_descriptions
from unified_kg_rag.domain.ingestion.entity_grounding import (
    is_grounded,
    is_name_grounded,
)
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    ModelPurpose,
    ModelRole,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.domain.prompts import GraphRefinementPrompt
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import (
    clean_display_name,
    ensure_list,
    entity_key,
)

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
    # The failed units whose error was their own (item_error_predicate).
    item_failed_text_unit_ids: list[str] = Field(default_factory=list)
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
    """Refine each text unit's extraction record with the model.

    Gleaning works on one record at a time (see ``TextUnitExtraction``): a
    unit's prompt lists its own entities and relationships, and its answer's
    additions and corrections edit its own record only. A correction never
    reaches what another unit says, so the graph a full build merges from the
    records is the one incremental runs, extracting the units in other runs,
    merge too.
    """

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
        # The renames the answer being applied asks for: entity id -> new
        # name, applied after its other issues (see _apply_answer), and the
        # new name's key -> the old name's key, so an issue naming either
        # finds the entity.
        self._renames: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        self._dropped_rejected_endpoint = 0
        self._dropped_low_confidence_entities = 0
        self._dropped_ungrounded_endpoint_name = 0

        self.factory = self.providers.llm_factory
        self.batch_processor = BatchProcessor.from_config(
            self.config,
            is_transient_error=transient_error_predicate(self.factory),
        )
        self.is_item_error = item_error_predicate(self.factory)
        self._item_failed_unit_ids: set[str] = set()

        self.max_entities_per_prompt = self.gleaning_config.max_entities_per_prompt
        self.max_relationships_per_prompt = (
            self.gleaning_config.max_relationships_per_prompt
        )

        robust_xml_output_parser = create_robust_xml_output_parser(
            model_purpose=ModelPurpose.INGESTION,
            factory=self.factory,
            enable_output_fixing=self.config.fixing.enabled,
            output_fixing_model_id=self.config.model_id(ModelRole.FIXING),
            min_output_tokens=GraphRefinementPrompt.output_floor(self.config),
            output_tags=["refinement_plan"],
        )
        self.graph_refiner = setup_chain(
            model_purpose=ModelPurpose.INGESTION,
            factory=self.factory,
            model_id=self.config.model_id(ModelRole.GLEANING),
            prompt_class=GraphRefinementPrompt,
            min_output_tokens=GraphRefinementPrompt.output_floor(self.config),
            parser=robust_xml_output_parser,
            custom_prompts=self.config.custom_prompts,
        )

    def glean(
        self, text_units: list[TextUnit], records: list[TextUnitExtraction]
    ) -> tuple[list[TextUnitExtraction], GleaningStats]:
        """Glean each record not yet gleaned for at most ``max_rounds`` rounds.

        Round 1 sends every such unit. Each later round re-sends only the
        units whose previous answer added an entity or relationship their
        record did not have. A unit stops once its answer adds nothing new:
        an empty ``<identified_issues>`` is the model saying nothing more is
        missing (the answer MS GraphRAG gets from its separate Y/N loop
        prompt), and items the record already has or that fail grounding are
        not new. A unit without a record (its extraction failed) is not
        gleaned.

        Returns copies of ``records``, the gleaned ones marked ``gleaned``
        (not one whose refinement failed in some round).
        """
        start_time = time.time()
        records = [record.model_copy(deep=True) for record in records]
        by_unit = {record.text_unit_id: record for record in records}
        stats = GleaningStats()
        self._failed_unit_ids = []
        self._item_failed_unit_ids = set()
        self._dropped_rejected_endpoint = 0
        self._dropped_low_confidence_entities = 0
        self._dropped_ungrounded_endpoint_name = 0
        max_rounds = self.gleaning_config.max_rounds

        units_to_glean = [
            unit
            for unit in text_units
            if unit.id in by_unit and not by_unit[unit.id].gleaned
        ]
        attempted = list(units_to_glean)
        logger.info("Starting graph gleaning of %s text units", len(units_to_glean))

        for round_num in range(1, max_rounds + 1):
            if not units_to_glean:
                break
            logger.info(
                "Starting gleaning round %s/%s (%s text units)",
                round_num,
                max_rounds,
                len(units_to_glean),
            )
            round_info, gained = self._perform_gleaning_round(
                units_to_glean, by_unit, round_num
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

        failed = set(self._failed_unit_ids)
        for unit in attempted:
            if unit.id not in failed:
                by_unit[unit.id].gleaned = True

        stats.total_rounds = len(stats.rounds)
        stats.num_failed_units = len(self._failed_unit_ids)
        stats.failed_text_unit_ids = sorted(failed)
        stats.item_failed_text_unit_ids = sorted(self._item_failed_unit_ids)
        stats.relationships_dropped_rejected_endpoint = self._dropped_rejected_endpoint
        stats.entities_dropped_low_confidence = self._dropped_low_confidence_entities
        stats.relationships_dropped_ungrounded_endpoint_name = (
            self._dropped_ungrounded_endpoint_name
        )
        stats.total_processing_time = time.time() - start_time

        self._log_completion_summary(stats, units_still_gaining=len(units_to_glean))

        return records, stats

    def _perform_gleaning_round(
        self,
        text_units: list[TextUnit],
        by_unit: dict[str, TextUnitExtraction],
        round_num: int,
    ) -> tuple[GleaningRound, set[str]]:
        """Send each unit to the model once and apply the answers to its record.

        Returns the round's stats and the ids of the units whose answer added
        an entity or relationship their record did not have.
        """
        round_start_time = time.time()
        records = [by_unit[unit.id] for unit in text_units]
        entities_before = sum(len(r.entities) for r in records)
        relationships_before = sum(len(r.relationships) for r in records)
        config_for_task = {
            "max_entities_per_prompt": self.max_entities_per_prompt,
            "max_relationships_per_prompt": self.max_relationships_per_prompt,
            "target_language": self.config.processing.translation.target_language.value,
        }
        unit_to_input: dict[str, Any] = {}
        for unit in text_units:
            record = by_unit[unit.id]
            try:
                unit_to_input[unit.id] = prepare_input(
                    unit,
                    unit_entities(record),
                    record.relationships,
                    config_for_task,
                )
            except Exception as e:
                logger.error(
                    "Error preparing gleaning input for unit '%s': %s", unit.id, e
                )

        # Only the units whose input prep succeeded are processed, and
        # execute_with_fallback returns results 1:1 with THOSE inputs, so the
        # zip below is aligned to them: a failed prep skips its unit rather
        # than misaligning every later answer.
        prepared_units = [u for u in text_units if u.id in unit_to_input]
        self._failed_unit_ids.extend(
            u.id for u in text_units if u.id not in unit_to_input
        )

        gained: set[str] = set()
        errors: dict[int, BaseException] = {}
        try:
            results = self.batch_processor.execute_with_fallback(
                items_to_process=prepared_units,
                prepare_inputs_func=lambda units: [unit_to_input[u.id] for u in units],
                sequential_func=self.graph_refiner.invoke,
                task_name="Graph Refinement",
                show_progress=self.show_progress,
                errors=errors,
            )
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.error("Error during graph refinement: %s", e)
            self._failed_unit_ids.extend(u.id for u in prepared_units)
            results = [BATCH_ITEM_FAILED] * len(prepared_units)
        self._item_failed_unit_ids.update(
            prepared_units[idx].id
            for idx in item_failed_indices(
                errors,
                prepared_units,
                self.is_item_error,
                "Graph refinement",
                self.batch_processor,
                lambda units: [
                    prepare_input(unit, [], [], config_for_task) for unit in units
                ],
                self.graph_refiner.invoke,
            )
        )

        for unit, result_data in zip(prepared_units, results, strict=True):
            if result_data is BATCH_ITEM_FAILED:
                if unit.id not in self._failed_unit_ids:
                    self._failed_unit_ids.append(unit.id)
                continue
            if self._apply_answer(
                result_data.get("refinement_plan", {}), unit, by_unit[unit.id]
            ):
                gained.add(unit.id)

        round_info = GleaningRound(
            round_number=round_num,
            units_gleaned=len(text_units),
            units_gained=len(gained),
            entities_before=entities_before,
            relationships_before=relationships_before,
            # Clamped: a correction can merge two of a unit's items into one.
            entities_added=max(
                0, sum(len(r.entities) for r in records) - entities_before
            ),
            relationships_added=max(
                0, sum(len(r.relationships) for r in records) - relationships_before
            ),
            processing_time=time.time() - round_start_time,
        )
        return round_info, gained

    def _apply_answer(
        self,
        result_data: dict[str, Any] | list[Any],
        unit: TextUnit,
        record: TextUnitExtraction,
    ) -> bool:
        """Apply one refinement answer to the unit's record.

        Returns whether it added an entity or relationship the record did not
        have (an entity named before only inside one of its relationships is
        not new).
        """
        if not result_data:
            logger.debug("No result data for text unit '%s'", unit.id)
            return False
        plan: dict[str, Any] | None = None
        if isinstance(result_data, dict):
            plan = result_data
        elif isinstance(result_data[0], dict):
            plan = result_data[0]
        if plan is None:
            logger.warning(
                "Could not extract a valid dictionary-based plan for unit "
                "'%s'. Received data type: %s",
                unit.id,
                type(result_data).__name__,
            )
            logger.debug("Refinement output preview: %s", str(result_data)[:250])
            return False

        known_keys = {entity_key(e.name) for e in unit_entities(record)}
        known_relationships = {r.id for r in record.relationships}
        self._renames, self._aliases = {}, {}
        try:
            issues_data = plan.get("identified_issues", {})
            if isinstance(issues_data, dict):
                issues = ensure_list(issues_data.get("issue", []))
            else:
                issues = ensure_list(issues_data)

            new_entities: list[Entity] = []
            new_relationships: list[Relationship] = []
            penalized: list[Relationship] = []
            for issue in issues:
                if not isinstance(issue, dict):
                    # <issue>none</issue>: nothing to apply.
                    continue
                self._process_issue(
                    issue, unit, record, new_entities, new_relationships, penalized
                )
            listed = unit_entities(record) + new_entities
            new_relationships += self._keep_penalized_relationships(
                penalized, unit, listed
            )
            new_relationships = self._drop_unsupported_relationships(
                new_relationships, unit, listed, record
            )
            self._add_to_record(record, new_entities, new_relationships)
            # Every issue of the answer names entities as its prompt listed
            # them, so the renames it asks for apply last (to its additions
            # too).
            for entity_id, name in self._renames.items():
                self._rename(record, entity_id, name)
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
            return False

        return any(entity_key(e.name) not in known_keys for e in new_entities) or any(
            r.id not in known_relationships and r.source_id != r.target_id
            for r in new_relationships
        )

    def _process_issue(
        self,
        issue: dict[str, Any],
        unit: TextUnit,
        record: TextUnitExtraction,
        new_entities: list[Entity],
        new_relationships: list[Relationship],
        penalized_relationships: list[Relationship],
    ) -> None:
        """Apply one refinement issue to the unit's record or its additions.

        An issue whose ``text_evidence`` the grounding check does not find in
        the chunk follows ``entity_grounding.action`` as at extraction. Under
        ``drop`` (and for a correction under either action) it is ignored,
        and the name key of a rejected MISSING_ENTITY is recorded as rejected
        for the unit, so a relationship naming it is dropped (see
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
        if not isinstance(details, dict):
            # <details>none</details>: no fields to apply.
            details = {}
        # coerce: an empty <issue_type/> parses to {} (no .upper()).
        issue_type = coerce_llm_text(issue.get("issue_type")).upper()
        grounding = self.extraction_config.entity_grounding
        penalize = False

        # The refinement prompt requires a <text_evidence> exact quote per issue
        # (sibling of <details>). When grounding is enabled, reject additions
        # whose evidence is not actually in the chunk — the same hallucination
        # guard applied at first-pass extraction.
        if grounding.enabled:
            text_evidence = issue.get("text_evidence")
            grounded = is_grounded(
                text_evidence if isinstance(text_evidence, str) else None,
                self.get_text_for_processing(unit),
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
                if issue_type == "MISSING_ENTITY":
                    name = clean_display_name(coerce_llm_text(details.get("name")))
                    if name:
                        record.reject(entity_key(name), "ungrounded")
                return

        if issue_type == "MISSING_ENTITY":
            entity = self.parse_entity_data(details, unit)
            if entity and penalize:
                base = entity.confidence if entity.confidence is not None else 1.0
                entity.confidence = base * grounding.penalty_factor
            if entity and self._below_confidence_threshold(entity, unit, record):
                return
            if entity:
                # Strip the reserved grounding span (gleaner already verified it).
                (entity.attributes or {}).pop("_source_text", None)
                new_entities.append(entity)
        elif issue_type == "MISSING_RELATIONSHIP":
            rel = self.parse_relationship_data(
                details, unit, self.build_entity_key_index(record.entities)
            )
            if rel:
                (rel.attributes or {}).pop("_source_text", None)
                if penalize:
                    rel.weight = (rel.weight or 1.0) * grounding.penalty_factor
                    penalized_relationships.append(rel)
                else:
                    new_relationships.append(rel)
        elif issue_type == "ENTITY_CORRECTION":
            self._apply_entity_correction(details, record, unit)
        elif issue_type == "RELATIONSHIP_CORRECTION":
            self._apply_relationship_correction(details, record, unit)
        else:
            # Every type the refinement prompt asks for must have a branch here.
            logger.warning(
                "Ignoring gleaner issue of unhandled type '%s' in chunk '%s'",
                issue_type or "<missing>",
                unit.short_id,
            )

    @staticmethod
    def _add_to_record(
        record: TextUnitExtraction,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> None:
        """Add an answer's entities and relationships to the unit's record.

        One the record already has (same identity key or id) keeps its type,
        confidence and strength and gains the proposed description and
        document filters: the model re-stating what it was shown is not a
        second statement of it. A self-loop is dropped.
        """
        by_key = {entity_key(e.name): e for e in record.entities}
        for entity in entities:
            existing = by_key.get(entity_key(entity.name))
            if existing is None:
                by_key[entity_key(entity.name)] = entity
                record.entities.append(entity)
                continue
            existing.description = merge_descriptions(
                (existing.description, entity.description)
            )
            existing.attributes = union_document_filters(
                existing.attributes, entity.attributes
            )
        record.relationships = _dedupe_relationships(
            record.relationships + relationships
        )

    @staticmethod
    def _keep_penalized_relationships(
        relationships: list[Relationship], unit: TextUnit, listed: list[Entity]
    ) -> list[Relationship]:
        """The penalized (ungrounded) relationships whose endpoints cite ``unit``.

        Extraction's ``penalize`` rule: an ungrounded relationship is kept
        only when the chunk lists both endpoints, since keeping one would make
        an endpoint the chunk did not list cite it (see
        ``GraphExtractor._apply_relationship_grounding``). ``listed`` are the
        entities citing the unit, its answer's additions included, so the
        order of the issues does not matter.
        """
        if not relationships:
            return []
        listed_ids = {entity.id for entity in listed}
        kept: list[Relationship] = []
        for rel in relationships:
            if {rel.source_id, rel.target_id} <= listed_ids:
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
        record: TextUnitExtraction,
    ) -> list[Relationship]:
        """Drop gleaned relationships whose endpoint the unit does not support.

        ``entities`` are the entities citing ``unit``, the answer's additions
        included.

        Every kept relationship makes its endpoints cite its unit, adding a
        stub for one no unit lists (:func:`cite_relationship_endpoints`), so
        a relationship naming an entity rejected for this unit would bring it
        back. Rejected means dropped as ungrounded by extraction for this
        unit, or proposed by gleaning (this or an earlier round) with
        evidence not found in it; the grounding check on the relationship
        covers only its quote, which need not name its endpoints. An
        ungrounded rejection is lifted once an entity of that name cites the
        unit (the unit listed it with grounded evidence). Extraction drops the
        same relationships (``_drop_relationships_to_dropped_entities``).

        An entity the confidence threshold removed for this unit is rejected
        whatever else cites the unit, so a stub never resurrects it.

        With entity grounding on, an endpoint no entity citing the unit names
        must also be named in the unit's text. With grounding off nothing is
        rejected and every relationship is kept.
        """
        if not relationships:
            return relationships
        grounding = self.extraction_config.entity_grounding
        cited = {entity_key(e.name) for e in entities}
        ungrounded = {k for k, why in record.rejected.items() if why == "ungrounded"}
        low_confidence = {
            k for k, why in record.rejected.items() if why == "low_confidence"
        }
        blocked = (ungrounded - cited) | low_confidence
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

    def _below_confidence_threshold(
        self, entity: Entity, unit: TextUnit, record: TextUnitExtraction
    ) -> bool:
        """Whether a gleaned entity fails the extraction confidence threshold.

        True when its confidence is below ``entity_confidence_threshold`` or
        the threshold removed an entity of that name from ``unit`` at
        extraction (the gleaner's answer carries no score, so it cannot
        overturn the extraction's). Records the key for the unit, so a
        relationship naming the entity is dropped too.
        """
        key = entity_key(entity.name)
        threshold = self.extraction_config.entity_confidence_threshold
        confidence = entity.confidence if entity.confidence is not None else 1.0
        if record.rejected.get(key) != "low_confidence" and (
            threshold <= 0.0 or confidence >= threshold
        ):
            return False
        record.reject(key, "low_confidence")
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

    def _prompt_key(self, name: Any) -> str | None:
        """The key of ``name`` as the unit's prompt listed it: a name the
        answer renames something to stands for the listed name."""
        key = _name_key(name)
        return self._aliases.get(key, key) if key is not None else None

    def _rekey(self, rel: Relationship) -> None:
        """Re-derive ``rel``'s id from its (source, target, type), as for an
        edge extracted that way."""
        rel.id = self._generate_relationship_id(
            rel.source_name or "", rel.target_name or "", rel.type or ""
        )
        rel.short_id = rel.id[:8]

    def _rename(self, record: TextUnitExtraction, old_id: str, name: str) -> None:
        """Rename the unit's entity ``old_id`` (listed, or named only as an
        endpoint) to ``name``: its id and every endpoint naming it follow.
        When the unit lists an entity of that name (another spelling of the
        key), the two become that entity: its id and name."""
        listed = next(
            (
                e
                for e in record.entities
                if entity_key(e.name) == entity_key(name) and e.id != old_id
            ),
            None,
        )
        if listed is not None:
            name = listed.name
        new_id = listed.id if listed is not None else self._generate_entity_id(name)
        for entity in record.entities:
            if entity.id == old_id:
                entity.id, entity.short_id, entity.name = new_id, new_id[:8], name
                entity.name_embedding = None
        for rel in record.relationships:
            if old_id not in (rel.source_id, rel.target_id):
                continue
            if rel.source_id == old_id:
                rel.source_id, rel.source_name = new_id, name
            if rel.target_id == old_id:
                rel.target_id, rel.target_name = new_id, name
            self._rekey(rel)
        record.entities = _dedupe_entities(record.entities)
        record.relationships = _dedupe_relationships(record.relationships)

    def _apply_entity_correction(
        self, details: dict[str, Any], record: TextUnitExtraction, unit: TextUnit
    ) -> bool:
        """Apply an ENTITY_CORRECTION to the unit's entity it names.

        Only an entity citing the unit (one its prompt lists) matches. A
        rename gives the entity the id of its new name, as extraction would,
        and moves the unit's relationships naming it along, once the answer's
        other issues are applied (they name the entity as the prompt did, or
        by its new name); the type and
        description of an entity the unit names only inside a relationship
        are not the unit's to correct (it states none). The prompt lists
        entity names only, so the model never sees the current description:
        a proposed one is merged into it rather than replacing it.

        Returns:
            True when at least one field changed.
        """
        name = self._clean_text(details.get("name"))
        key = self._prompt_key(name)
        current = next(
            (e for e in unit_entities(record) if entity_key(e.name) == key), None
        )
        if current is None:
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
        entity_id = current.id
        if corrected_name and corrected_name != current.name:
            self._renames[entity_id] = corrected_name
            self._aliases[entity_key(corrected_name)] = entity_key(current.name)
            corrected.append("name")

        entity = next((e for e in record.entities if e.id == entity_id), None)
        corrected_type = self._clean_text(details.get("corrected_type"))
        description = self._clean_text(details.get("description"))
        if entity is not None:
            if corrected_type and corrected_type != entity.type:
                entity.type = corrected_type
                corrected.append("type")
            merged = merge_descriptions((entity.description, description))
            if merged != entity.description:
                entity.description = merged
                entity.description_embedding = None
                corrected.append("description")
        elif corrected_type or description:
            logger.debug(
                "ENTITY_CORRECTION in chunk '%s' names an entity the chunk "
                "states only in a relationship; keeping its type and "
                "description",
                unit.short_id,
            )

        if not corrected:
            logger.debug(
                "ENTITY_CORRECTION for '%s' in chunk '%s' proposed no change",
                current.name,
                unit.short_id,
            )
            return False
        if entity is not None:
            entity.updated_at = datetime.now()
        logger.info(
            "Applied ENTITY_CORRECTION to entity '%s' in chunk '%s' (%s)",
            entity_id[:8],
            unit.short_id,
            ", ".join(corrected),
        )
        return True

    def _apply_relationship_correction(
        self, details: dict[str, Any], record: TextUnitExtraction, unit: TextUnit
    ) -> bool:
        """Apply a RELATIONSHIP_CORRECTION to the unit's relationship it names.

        Scope matches what the prompt asks for: a wrong type, a reversed
        direction, or a wrong description. Re-pointing an edge at a different
        pair of entities is not a correction and is rejected — that is a
        MISSING_RELATIONSHIP plus a deletion, which this stage cannot express.
        A retyped or reversed relationship takes the id of its corrected
        (source, target, type), as extraction would give it; it merges with
        one the unit already states that way. A proposed description is
        merged into the current one, which the prompt does not show.

        Returns:
            True when at least one field changed.
        """
        pair = self._prompt_key(details.get("source")), self._prompt_key(
            details.get("target")
        )
        candidates = [
            rel
            for rel in record.relationships
            if None not in pair
            and (entity_key(rel.source_name or ""), entity_key(rel.target_name or ""))
            == pair
        ]
        relationship = self._find_relationship(candidates, details.get("type"))
        if relationship is None:
            logger.info(
                "Dropping RELATIONSHIP_CORRECTION in chunk '%s': no single current "
                "relationship matches its endpoints",
                unit.short_id,
            )
            return False

        original_short_id = relationship.short_id
        corrected: list[str] = []
        corrected_type = self._clean_text(details.get("corrected_type"))
        if corrected_type and corrected_type != relationship.type:
            relationship.type = corrected_type
            corrected.append("type")
        # A reversal is the corrected endpoints naming the current pair the other
        # way round. Ids and names move together.
        if (
            self._prompt_key(details.get("corrected_source")),
            self._prompt_key(details.get("corrected_target")),
        ) == (
            entity_key(relationship.target_name or ""),
            entity_key(relationship.source_name or ""),
        ):
            relationship.source_id, relationship.target_id = (
                relationship.target_id,
                relationship.source_id,
            )
            relationship.source_name, relationship.target_name = (
                relationship.target_name,
                relationship.source_name,
            )
            corrected.append("direction")
        if corrected:
            self._rekey(relationship)

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
        record.relationships = _dedupe_relationships(record.relationships)
        logger.info(
            "Applied RELATIONSHIP_CORRECTION to '%s' in chunk '%s' (%s)",
            original_short_id,
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


def _dedupe_entities(entities: list[Entity]) -> list[Entity]:
    """One text unit's entities, one per id: a rename onto a name the unit
    also lists makes them one entity, the first keeping its type and gaining
    the other's description, filters and (max) confidence."""
    kept: dict[str, Entity] = {}
    for entity in entities:
        existing = kept.get(entity.id)
        if existing is None:
            kept[entity.id] = entity
            continue
        existing.description = merge_descriptions(
            (existing.description, entity.description)
        )
        existing.attributes = union_document_filters(
            existing.attributes, entity.attributes
        )
        existing.confidence = max(
            existing.confidence if existing.confidence is not None else 0.0,
            entity.confidence if entity.confidence is not None else 0.0,
        )
    return list(kept.values())


def _dedupe_relationships(relationships: list[Relationship]) -> list[Relationship]:
    """One text unit's relationships, one per id, without self-loops.

    Within a unit a relationship stated again (a gleaned re-proposal, or a
    correction turning one into another) is the same statement: the first
    keeps its strength and gains the other's description and document
    filters.
    """
    kept: dict[str, Relationship] = {}
    for rel in relationships:
        if rel.source_id == rel.target_id:
            continue
        existing = kept.get(rel.id)
        if existing is None:
            kept[rel.id] = rel
            continue
        existing.description = merge_descriptions(
            (existing.description, rel.description)
        )
        existing.attributes = union_document_filters(
            existing.attributes, rel.attributes
        )
    return list(kept.values())
