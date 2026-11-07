# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-aware evaluator: entity/relationship coverage of generated answers.

Consumes the previously-unused ``EvaluationGroundTruth.expected_entities`` and
``expected_relationships`` fields. For each query it measures how many of the
expected graph artifacts the generated answer actually surfaces (whole-word
match after SQuAD normalization, shared with the answer-match evaluator via
``text_matching.phrase_in_text``; Korean particles tolerated; substring match
for single-word CJK phrases), reporting
coverage (= recall) — a deterministic, LLM-free
signal complementing the LangChain/RAGAS text-similarity scores. Precision/F1 are
deliberately NOT reported: they would require enumerating every entity in a
free-text answer (not reliably possible), so emitting them would only duplicate
the recall signal under another name.

An expected relationship given as a ``{"source": A, "target": B}`` pair or an
``"A -> B"`` string counts as covered when the answer mentions both endpoints
(answers rarely restate a relation verbatim); any other string must appear as
a phrase.

Answer-text coverage grows with answer length (a wordier answer names more
things), so the evaluator also reports ``retrieved_entity_coverage`` /
``retrieved_relationship_coverage`` — the same expectations matched against
the entity names and relationship endpoints of the sources the answer model
was given, by entity key (``compact_name_key``: the key the resolver links
spellings of one entity by, so "ACME Corp." is "Acme", but "Seoul" is not
"Seoul National University") — and ``answer_length`` (characters)
alongside. Expected entities that share that key are one expectation, met
when the answer mentions any of their spellings. Retrieved coverage is
skipped for a query whose context has no entity (or relationship) source, not
scored 0: a strategy that reads no such sources by design (``global``) has none
to measure. A strategy that finds none for a query is therefore not scored on
it either; a comparison reports those queries as ``unpaired``. When the
guardrail blocked the answer (dropping its sources), retrieved coverage is
skipped as ``guardrail_blocked``.

Expected artifacts are threaded onto ``EvaluationResult.metadata`` by the
manager (keys ``expected_entities`` / ``expected_relationships``), so this
evaluator needs no signature change to the abstract base.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from unified_kg_rag.adapters.evaluators.base import (
    SKIP_REASON_GUARDRAIL_BLOCKED,
    SKIPPED_METRICS_KEY,
    BaseGraphRAGEvaluator,
)
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.domain.ingestion.base_resolver import compact_name_key
from unified_kg_rag.domain.models import (
    Config,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
)
from unified_kg_rag.domain.retrieval.citations import strip_markers
from unified_kg_rag.shared import get_logger

from .text_matching import phrase_in_text

logger = get_logger(__name__)


class GraphAwareEvaluator(BaseGraphRAGEvaluator):
    """Scores entity/relationship coverage of the generated answer."""

    def __init__(self, config: Config, **kwargs: Any):
        super().__init__(config, EvaluatorType.GRAPH_AWARE, **kwargs)

    def metric_types(self) -> list[EvaluationMetricType]:
        return [
            EvaluationMetricType.ENTITY_COVERAGE,
            EvaluationMetricType.RELATIONSHIP_COVERAGE,
            EvaluationMetricType.RETRIEVED_ENTITY_COVERAGE,
            EvaluationMetricType.RETRIEVED_RELATIONSHIP_COVERAGE,
            EvaluationMetricType.ANSWER_LENGTH,
        ]

    @staticmethod
    def _relationship_endpoints(item: Any) -> tuple[str, str] | None:
        """(source, target) for a pair dict or an ``"A -> B"`` string, else None."""
        if isinstance(item, dict):
            return str(item.get("source", "")), str(item.get("target", ""))
        if isinstance(item, str) and item.count("->") == 1:
            source, target = (part.strip() for part in item.split("->"))
            if source and target:
                return source, target
        return None

    @classmethod
    def _is_covered(cls, item: Any, answer: str, relationships: bool) -> bool:
        if not item:
            return False
        endpoints = cls._relationship_endpoints(item) if relationships else None
        if endpoints is not None:
            return all(phrase_in_text(end, answer) for end in endpoints)
        return isinstance(item, str) and phrase_in_text(item, answer)

    @classmethod
    def _coverage(
        cls, expected: list[Any], answer: str, relationships: bool = False
    ) -> tuple[float | None, int]:
        """Return (coverage, num_matched) for expected-artifacts-in-answer.

        Coverage = matched / expected (i.e. recall): the fraction of expected
        graph artifacts whose full word sequence appears, word-boundary aware,
        in the answer (so "AI" does not match inside "airport"). We intentionally
        report ONLY coverage — not precision/F1 — because precision would require
        enumerating the answer's own entities/relationships, which we cannot do
        from free text; emitting precision as a copy of recall (the previous
        behaviour) overstated the signal.

        Returns ``None`` when nothing is expected for the dimension, so a query
        with (say) only expected entities is not scored 0 for having no expected
        relationships (the metric is skipped instead). With
        ``relationships=True``, pair/arrow items count when both endpoints match.
        """
        if not expected:
            return None, 0
        matched = sum(
            1 for item in expected if cls._is_covered(item, answer, relationships)
        )
        return matched / len(expected), matched

    @staticmethod
    def _entity_groups(expected: list[Any]) -> dict[str, list[Any]]:
        """The expected entities by entity key, each with its spellings in
        dataset order: "Acme Corp" and "ACME Corp." are one expectation."""
        groups: dict[str, list[Any]] = {}
        for item in expected:
            key = compact_name_key(item) if isinstance(item, str) else repr(item)
            groups.setdefault(key, []).append(item)
        return groups

    @classmethod
    def _entity_coverage(
        cls, groups: dict[str, list[Any]], answer: str
    ) -> tuple[float | None, int]:
        """``_coverage`` of the entity expectations: one is met when the
        answer mentions any of its spellings."""
        if not groups:
            return None, 0
        matched = sum(
            1
            for spellings in groups.values()
            if any(cls._is_covered(item, answer, False) for item in spellings)
        )
        return matched / len(groups), matched

    @staticmethod
    def _retrieved_entity_coverage(
        groups: dict[str, list[Any]], retrieved: list[str]
    ) -> tuple[float | None, int]:
        """Coverage of the entity expectations over the retrieved entity
        sources: met when a retrieved entity has the same entity key."""
        if not groups:
            return None, 0
        names = {compact_name_key(name) for name in retrieved if name}
        matched = sum(
            1
            for key, spellings in groups.items()
            if key and isinstance(spellings[0], str) and key in names
        )
        return matched / len(groups), matched

    @classmethod
    def _retrieved_coverage(
        cls, expected: list[Any], retrieved: list[dict[str, str]]
    ) -> tuple[float | None, int]:
        """Coverage of the expected relationships over the retrieved ones.

        An expected relationship pair is covered when ONE retrieved
        relationship has both endpoints, in either direction, by entity key;
        any other relationship string must appear in a retrieved
        relationship's endpoints or description.
        """
        if not expected:
            return None, 0

        def covered(item: Any) -> bool:
            if not item:
                return False
            for element in retrieved:
                source = element.get("source", "")
                target = element.get("target", "")
                endpoints = cls._relationship_endpoints(item)
                if endpoints is not None:
                    keys = tuple(compact_name_key(end) for end in endpoints)
                    ends = (compact_name_key(source), compact_name_key(target))
                    if all(keys) and ends in (keys, keys[::-1]):
                        return True
                elif isinstance(item, str) and phrase_in_text(
                    item, " ".join((source, target, element.get("description", "")))
                ):
                    return True
            return False

        matched = sum(1 for item in expected if covered(item))
        return matched / len(expected), matched

    def _build_metrics(
        self, result: EvaluationResult
    ) -> tuple[list[EvaluationMetric], dict[str, Any]]:
        answer = result.generated_answer or ""
        if self.config.search.citations.enabled:
            # The answer's words: its citation markers ([1]) are no
            # coverage and no length (as answer_match reads it).
            answer = strip_markers(answer)
        expected_entities = result.metadata.get("expected_entities", []) or []
        expected_relationships = result.metadata.get("expected_relationships", []) or []
        source_types = set(result.retrieved_source_types)

        entity_groups = self._entity_groups(expected_entities)
        e_cov, e_matched = self._entity_coverage(entity_groups, answer)
        r_cov, r_matched = self._coverage(
            expected_relationships, answer, relationships=True
        )
        re_cov, re_matched = self._retrieved_entity_coverage(
            entity_groups, result.retrieved_entities
        )
        rr_cov, rr_matched = self._retrieved_coverage(
            expected_relationships, result.retrieved_relationships
        )

        # Only emit metrics for dimensions that actually have expectations; a
        # missing dimension is recorded as skipped, not scored 0. Retrieved
        # coverage is also skipped when the context had no source of that type
        # (global search reads community reports only): it is undefined there,
        # not a miss.
        skipped: dict[str, str] = {}
        blocked = result.guardrail_blocked == "output"
        if blocked or SectionType.ENTITY.value not in source_types:
            re_cov = None
            if expected_entities:
                skipped[EvaluationMetricType.RETRIEVED_ENTITY_COVERAGE.value] = (
                    SKIP_REASON_GUARDRAIL_BLOCKED if blocked else "no_entity_sources"
                )
        if blocked or SectionType.RELATIONSHIP.value not in source_types:
            rr_cov = None
            if expected_relationships:
                skipped[EvaluationMetricType.RETRIEVED_RELATIONSHIP_COVERAGE.value] = (
                    SKIP_REASON_GUARDRAIL_BLOCKED
                    if blocked
                    else "no_relationship_sources"
                )
        values = {
            EvaluationMetricType.ENTITY_COVERAGE: e_cov,
            EvaluationMetricType.RELATIONSHIP_COVERAGE: r_cov,
            EvaluationMetricType.RETRIEVED_ENTITY_COVERAGE: re_cov,
            EvaluationMetricType.RETRIEVED_RELATIONSHIP_COVERAGE: rr_cov,
        }
        metrics = [
            EvaluationMetric(metric_type=metric_type, value=value)
            for metric_type, value in values.items()
            if value is not None
        ]
        # Answer-text coverage rises with answer length; report the length next
        # to it so a wordier strategy's coverage gain is visible as such.
        if e_cov is not None or r_cov is not None:
            metrics.append(
                EvaluationMetric(
                    metric_type=EvaluationMetricType.ANSWER_LENGTH,
                    value=float(len(" ".join(answer.split()))),
                )
            )
        metadata = {
            **self._extract_search_metadata(result),
            # Distinct by entity key (spellings of one entity count once).
            "expected_entity_count": len(entity_groups),
            "matched_entity_count": e_matched,
            "retrieved_matched_entity_count": re_matched,
            "expected_relationship_count": len(expected_relationships),
            "matched_relationship_count": r_matched,
            "retrieved_matched_relationship_count": rr_matched,
        }
        if not expected_entities:
            for metric_type in (
                EvaluationMetricType.ENTITY_COVERAGE,
                EvaluationMetricType.RETRIEVED_ENTITY_COVERAGE,
            ):
                skipped[metric_type.value] = "no_expected_entities"
        if not expected_relationships:
            for metric_type in (
                EvaluationMetricType.RELATIONSHIP_COVERAGE,
                EvaluationMetricType.RETRIEVED_RELATIONSHIP_COVERAGE,
            ):
                skipped[metric_type.value] = "no_expected_relationships"
        if not expected_entities and not expected_relationships:
            skipped[EvaluationMetricType.ANSWER_LENGTH.value] = "no_expected_artifacts"
        if skipped:
            metadata[SKIPPED_METRICS_KEY] = skipped
        return metrics, metadata

    def evaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        metrics, metadata = self._build_metrics(result)
        return EvaluationReport(
            query_id=query.query_id,
            evaluator_type=self.evaluator_type,
            metrics=metrics,
            evaluation_time=datetime.now(),
            metadata=metadata,
        )
