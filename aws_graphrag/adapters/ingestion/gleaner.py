# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any

import boto3
from pydantic import BaseModel, Field
from tqdm import tqdm

from aws_graphrag.adapters.aws import BedrockLanguageModelFactory
from aws_graphrag.adapters.aws.chain_factory import (
    create_robust_xml_output_parser,
    setup_chain,
)
from aws_graphrag.domain.ingestion.base_processor import (
    BaseProcessor,
    check_entity_relevance_task,
    check_relationship_relevance_task,
)
from aws_graphrag.domain.models import Config, Entity, Relationship, TextUnit
from aws_graphrag.domain.prompts import GraphRefinementPrompt
from aws_graphrag.shared import get_logger
from aws_graphrag.shared.utils import (
    BatchProcessor,
    default_max_workers,
    ensure_list,
)

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
    entities_before: int
    relationships_before: int
    entities_added: int
    relationships_added: int
    quality_improvement: float
    convergence_score: float
    processing_time: float


class GleaningStats(BaseModel):
    total_rounds: int = 0
    total_entities_added: int = 0
    total_relationships_added: int = 0
    initial_quality_score: float = 0.0
    final_quality_score: float = 0.0
    total_processing_time: float = 0.0
    convergence_achieved: bool = False
    rounds: list[GleaningRound] = Field(default_factory=list)

    @property
    def quality_improvement(self) -> float:
        return self.final_quality_score - self.initial_quality_score

    @property
    def average_round_time(self) -> float:
        return (
            self.total_processing_time / self.total_rounds
            if self.total_rounds > 0
            else 0.0
        )

    @property
    def entities_per_round(self) -> float:
        return (
            self.total_entities_added / self.total_rounds
            if self.total_rounds > 0
            else 0.0
        )

    @property
    def relationships_per_round(self) -> float:
        return (
            self.total_relationships_added / self.total_rounds
            if self.total_rounds > 0
            else 0.0
        )


class GraphGleaner(BaseProcessor):
    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        max_workers: int | None = None,
        use_process_pool: bool = True,
        show_progress: bool = True,
    ) -> None:
        super().__init__(config)
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name
        )
        self.gleaning_config = self.config.processing.gleaning
        self.ignore_errors = self.config.processing.ignore_errors
        self.max_workers = max_workers or default_max_workers()
        self.use_process_pool = use_process_pool
        self.show_progress = show_progress

        self.factory = BedrockLanguageModelFactory(
            config=self.config,
            boto_session=self.boto_session,
            region_name=self.config.aws.bedrock.region_name,
        )
        self.batch_processor = BatchProcessor()

        self.max_entities_per_prompt = self.gleaning_config.max_entities_per_prompt
        self.max_relationships_per_prompt = (
            self.gleaning_config.max_relationships_per_prompt
        )

        robust_xml_output_parser = create_robust_xml_output_parser(
            factory=self.factory,
            enable_output_fixing=self.config.fixing.enabled,
            output_fixing_model_id=self.config.fixing.fixing_model_id,
        )
        self.graph_refiner = setup_chain(
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
        start_time = time.time()
        current_entities = initial_entities.copy()
        current_relationships = initial_relationships.copy()

        initial_quality = self._calculate_initial_quality(
            current_entities, current_relationships
        )
        stats = GleaningStats(initial_quality_score=initial_quality)

        logger.info(
            "Starting graph gleaning from %s text units, "
            "%s entities, "
            "%s relationships "
            "(initial quality: %.3f)",
            len(text_units),
            len(current_entities),
            len(current_relationships),
            initial_quality,
        )

        current_quality = initial_quality
        previous_quality = initial_quality

        for round_num in range(1, self.gleaning_config.max_rounds + 1):
            round_start_time = time.time()
            logger.info(
                "Starting gleaning round %s/%s",
                round_num,
                self.gleaning_config.max_rounds,
            )

            entities_before = len(current_entities)
            relationships_before = len(current_relationships)

            round_stats = self._perform_gleaning_round(
                text_units=text_units,
                current_entities=current_entities,
                current_relationships=current_relationships,
                round_num=round_num,
                entities_before=entities_before,
                relationships_before=relationships_before,
                previous_quality=previous_quality,
                round_start_time=round_start_time,
            )

            current_entities = round_stats["entities"]
            current_relationships = round_stats["relationships"]
            current_quality = round_stats["quality"]

            stats.rounds = stats.rounds + [round_stats["round_info"]]
            stats.total_entities_added += round_stats["round_info"].entities_added
            stats.total_relationships_added += round_stats[
                "round_info"
            ].relationships_added

            logger.info(
                "Round %s completed: "
                "+%s entities, "
                "+%s relationships, "
                "quality: %.3f "
                "(%+.3f), "
                "convergence: %.3f",
                round_num,
                round_stats["round_info"].entities_added,
                round_stats["round_info"].relationships_added,
                current_quality,
                round_stats["round_info"].quality_improvement,
                round_stats["round_info"].convergence_score,
            )

            if self._should_stop_gleaning(
                round_stats["round_info"].convergence_score,
                round_stats["round_info"].quality_improvement,
                current_quality,
            ):
                stats.convergence_achieved = True
                break

            previous_quality = current_quality

        stats.total_rounds = len(stats.rounds)
        stats.final_quality_score = current_quality
        stats.total_processing_time = time.time() - start_time

        self._log_completion_summary(stats)

        return current_entities, current_relationships, stats

    def _calculate_initial_quality(
        self, entities: list[Entity], relationships: list[Relationship]
    ) -> float:
        if not entities and not relationships:
            return 0.0
        entity_scale = self.gleaning_config.initial_quality_entity_scale
        rel_scale = self.gleaning_config.initial_quality_relationship_scale
        entity_completeness = min(0.5, len(entities) / entity_scale)
        relationship_completeness = min(0.5, len(relationships) / rel_scale)
        return (entity_completeness + relationship_completeness) / 2.0

    def _perform_gleaning_round(
        self,
        text_units: list[TextUnit],
        current_entities: list[Entity],
        current_relationships: list[Relationship],
        round_num: int,
        entities_before: int,
        relationships_before: int,
        previous_quality: float,
        round_start_time: float,
    ) -> dict[str, Any]:
        logger.debug(
            "Round %s: Starting LLM refinement for %s text units",
            round_num,
            len(text_units),
        )

        newly_discovered_entities, newly_discovered_relationships, quality_scores = (
            self._perform_llm_refinement(
                text_units, current_entities, current_relationships
            )
        )

        logger.debug(
            "Round %s: LLM refinement produced %s entities and %s relationships",
            round_num,
            len(newly_discovered_entities),
            len(newly_discovered_relationships),
        )

        combined_entities = current_entities + newly_discovered_entities
        combined_relationships = current_relationships + newly_discovered_relationships

        merged_entities, entity_id_map = self._merge_duplicate_entities(
            combined_entities
        )
        merged_relationships = self._update_relationships_after_merge(
            combined_relationships, {e.id for e in merged_entities}, entity_id_map
        )

        # Clamp to >= 0: a round that only discovers entities/relationships which
        # merge into existing ones (or whose edges are dropped as orphaned) can
        # make the post-merge count <= the prior count. A negative "added" value
        # would corrupt the convergence score (negative change_rate inflates
        # convergence toward 1.0 -> premature stop).
        entities_added = max(0, len(merged_entities) - entities_before)
        relationships_added = max(0, len(merged_relationships) - relationships_before)
        current_quality = self._calculate_graph_quality(quality_scores)
        quality_improvement = current_quality - previous_quality
        convergence_score = self._calculate_convergence_score(
            entities_added, relationships_added, quality_improvement
        )

        round_info = GleaningRound(
            round_number=round_num,
            entities_before=entities_before,
            relationships_before=relationships_before,
            entities_added=entities_added,
            relationships_added=relationships_added,
            quality_improvement=quality_improvement,
            convergence_score=convergence_score,
            processing_time=time.time() - round_start_time,
        )

        return {
            "entities": merged_entities,
            "relationships": merged_relationships,
            "quality": current_quality,
            "round_info": round_info,
        }

    def _perform_llm_refinement(
        self,
        text_units: list[TextUnit],
        current_entities: list[Entity],
        current_relationships: list[Relationship],
    ) -> tuple[list[Entity], list[Relationship], dict[str, float]]:
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
            ProcessPoolExecutor if self.use_process_pool else ThreadPoolExecutor
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
                disable=not self.show_progress,
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

        try:
            results = self.batch_processor.execute_with_fallback(
                items_to_process=text_units,
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
            return [], [], {}

        all_new_entities, all_new_relationships = [], []
        quality_scores_aggregator: dict[str, list[float]] = {
            "completeness": [],
            "accuracy": [],
        }

        for item, result_data in zip(text_units, results, strict=True):
            new_entities, new_relationships, quality_scores = (
                self._parse_refinement_output(
                    result_data.get("refinement_plan", {}), item, current_entities
                )
            )
            all_new_entities.extend(new_entities)
            all_new_relationships.extend(new_relationships)
            self._aggregate_quality_scores(quality_scores, quality_scores_aggregator)

        if all_new_entities:
            all_entity_details = [f"'{entity.name}'" for entity in all_new_entities]
            logger.debug("All new entities: %s", all_entity_details)
        if all_new_relationships:
            all_relationship_details = [
                f"'{rel.source_name}' -> '{rel.target_name}' (type: '{rel.type}')"
                for rel in all_new_relationships
            ]
            logger.debug("All new relationships: %s", all_relationship_details)

        avg_quality_scores = {
            "completeness": self._calculate_average(
                quality_scores_aggregator["completeness"]
            ),
            "accuracy": self._calculate_average(quality_scores_aggregator["accuracy"]),
        }
        return all_new_entities, all_new_relationships, avg_quality_scores

    def _calculate_graph_quality(self, quality_scores: dict[str, float]) -> float:
        completeness = quality_scores.get("completeness", 0.5)
        accuracy = quality_scores.get("accuracy", 0.5)
        completeness_weight = self.gleaning_config.quality_completeness_weight
        return (completeness * completeness_weight) + (
            accuracy * (1.0 - completeness_weight)
        )

    @staticmethod
    def _aggregate_quality_scores(
        quality_scores: dict[str, float], aggregator: dict[str, list[float]]
    ) -> None:
        if quality_scores.get("completeness") is not None:
            aggregator["completeness"].append(quality_scores["completeness"])
        if quality_scores.get("accuracy") is not None:
            aggregator["accuracy"].append(quality_scores["accuracy"])

    @staticmethod
    def _calculate_average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _parse_refinement_output(
        self,
        result_data: dict[str, Any] | list[Any],
        unit: TextUnit,
        existing_entities: list[Entity],
    ) -> tuple[list[Entity], list[Relationship], dict[str, float]]:
        new_entities: list[Entity] = []
        new_relationships: list[Relationship] = []
        quality_scores: dict[str, float] = {}

        if not result_data:
            logger.debug("No result data for text unit '%s'", unit.id)
            return new_entities, new_relationships, quality_scores

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
                return new_entities, new_relationships, quality_scores

            quality_scores = self._extract_quality_scores(plan)

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

        return new_entities, new_relationships, quality_scores

    @staticmethod
    def _extract_quality_scores(plan: dict[str, Any]) -> dict[str, float]:
        scores_data = plan.get("quality_scores", {})

        if isinstance(scores_data, list):
            merged_scores = {}
            for item in scores_data:
                if isinstance(item, dict):
                    merged_scores.update(item)
            scores_data = merged_scores

        if not isinstance(scores_data, dict):
            return {"completeness": 0.0, "accuracy": 0.0}

        try:
            completeness = float(scores_data.get("completeness_score", 0.0))
            accuracy = float(scores_data.get("accuracy_score", 0.0))
        except (ValueError, TypeError):
            completeness = 0.0
            accuracy = 0.0

        return {
            "completeness": completeness,
            "accuracy": accuracy,
        }

    def _process_issue(
        self,
        issue: dict[str, Any],
        unit: TextUnit,
        current_and_new_entities: list[Entity],
        new_entities: list[Entity],
        new_relationships: list[Relationship],
    ) -> None:
        details = issue.get("details", {})
        issue_type = issue.get("issue_type", "").upper()

        if issue_type == "MISSING_ENTITY":
            entity = self.parse_entity_data(details, unit)
            if entity:
                new_entities.append(entity)
                current_and_new_entities.append(entity)
        elif issue_type == "MISSING_RELATIONSHIP":
            entity_name_to_id = {
                entity.name: entity.id for entity in current_and_new_entities
            }
            rel = self.parse_relationship_data(details, unit, entity_name_to_id)
            if rel:
                new_relationships.append(rel)

    @staticmethod
    def _merge_duplicate_entities(
        entities: list[Entity],
    ) -> tuple[list[Entity], dict[str, str]]:
        entities_map: dict[str, Entity] = {}
        id_remap: dict[str, str] = {}
        type_counts: dict[str, dict[str, int]] = {}
        merged_names = []

        for entity in entities:
            key = entity.name.lower()
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
                key = (source_id, target_id, rel.type.lower() if rel.type else "")

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

                    existing_rel.weight = (existing_rel.weight or 1.0) + (
                        rel.weight or 1.0
                    )
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

    def _calculate_convergence_score(
        self,
        entities_added: int,
        relationships_added: int,
        quality_improvement: float,
    ) -> float:
        if entities_added == 0 and relationships_added == 0:
            return 1.0
        change_scale = self.gleaning_config.convergence_change_scale
        change_rate = (entities_added + relationships_added) / change_scale
        convergence = 1.0 - min(1.0, change_rate + abs(quality_improvement))
        return max(0.0, convergence)

    def _should_stop_gleaning(
        self,
        convergence_score: float,
        quality_improvement: float,
        current_quality: float,
    ) -> bool:
        if convergence_score >= self.gleaning_config.convergence_threshold:
            logger.info(
                "Convergence achieved: score %.3f >= threshold %.3f",
                convergence_score,
                self.gleaning_config.convergence_threshold,
            )
            return True

        if abs(quality_improvement) < self.gleaning_config.min_improvement_threshold:
            logger.info(
                "Quality improvement below threshold: %.3f < %.3f",
                quality_improvement,
                self.gleaning_config.min_improvement_threshold,
            )
            return True

        if current_quality >= self.gleaning_config.quality_threshold:
            logger.info(
                "Quality target reached: %.3f >= %.3f",
                current_quality,
                self.gleaning_config.quality_threshold,
            )
            return True

        return False

    @staticmethod
    def _log_completion_summary(stats: GleaningStats) -> None:
        logger.info(
            "Graph gleaning completed: %s rounds, "
            "%s entities added, "
            "%s relationships added, "
            "quality improved from %.3f to "
            "%.3f "
            "(%+.3f) in %.2fs",
            stats.total_rounds,
            stats.total_entities_added,
            stats.total_relationships_added,
            stats.initial_quality_score,
            stats.final_quality_score,
            stats.quality_improvement,
            stats.total_processing_time,
        )

        if stats.total_rounds > 0:
            logger.info(
                "Average per round: %.1f entities, "
                "%.1f relationships, "
                "%.2fs processing time",
                stats.entities_per_round,
                stats.relationships_per_round,
                stats.average_round_time,
            )

        if stats.convergence_achieved:
            logger.info("Gleaning process converged successfully")
        else:
            logger.warning(
                "Gleaning process did not converge after %s rounds", stats.total_rounds
            )
