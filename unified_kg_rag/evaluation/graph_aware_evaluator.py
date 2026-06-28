# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-aware evaluator: entity/relationship coverage of generated answers.

Consumes the previously-unused ``EvaluationGroundTruth.expected_entities`` and
``expected_relationships`` fields. For each query it measures how many of the
expected graph artifacts the generated answer actually surfaces (case-insensitive
substring match), reporting coverage (= recall) — a deterministic, LLM-free
signal complementing the LangChain/RAGAS text-similarity scores. Precision/F1 are
deliberately NOT reported: they would require enumerating every entity in a
free-text answer (not reliably possible), so emitting them would only duplicate
the recall signal under another name.

Expected artifacts are threaded onto ``EvaluationResult.metadata`` by the
manager (keys ``expected_entities`` / ``expected_relationships``), so this
evaluator needs no signature change to the abstract base.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from unified_kg_rag.domain.models import (
    Config,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
)
from unified_kg_rag.shared import get_logger

from .base import BaseGraphRAGEvaluator

logger = get_logger(__name__)


class GraphAwareEvaluator(BaseGraphRAGEvaluator):
    """Scores entity/relationship coverage of the generated answer."""

    def __init__(self, config: Config, rag_chain: Any | None = None, **kwargs: Any):
        super().__init__(
            config, EvaluatorType.GRAPH_AWARE, rag_chain=rag_chain, **kwargs
        )

    def _initialize_evaluator(self, **kwargs: Any) -> None:
        # Pure, deterministic evaluator — no model to initialize.
        pass

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase word tokens, stripping surrounding punctuation.

        Deterministic and regex-free: split on whitespace, then strip
        non-alphanumeric characters from each token's edges. This gives
        word-boundary matching so ``"AI"`` does not match inside ``"airport"``.
        """
        tokens = []
        for raw in text.lower().split():
            token = raw.strip("\"'.,;:!?()[]{}<>/\\|`")
            if token:
                tokens.append(token)
        return tokens

    @staticmethod
    def _is_spaceless_script(text: str) -> bool:
        """True if the text has CJK characters and no internal whitespace.

        Such scripts are not whitespace-tokenizable, so contiguous word-token
        matching would always miss; coverage must use a substring check instead.
        """
        if any(ch.isspace() for ch in text.strip()):
            return False
        return any(
            "぀" <= ch <= "鿿"  # Hiragana, Katakana, CJK ideographs
            or "가" <= ch <= "힣"  # Hangul syllables
            for ch in text
        )

    @classmethod
    def _phrase_in_tokens(
        cls, phrase: str, answer_tokens: list[str], answer_text: str
    ) -> bool:
        """True if the expected phrase appears in the answer.

        Latin/space-delimited scripts use contiguous word-token matching (so
        "AI" does not match inside "airport"). Space-less scripts (CJK) are not
        whitespace-tokenizable, so word matching would make coverage always 0 —
        for those, fall back to a normalized substring check on the raw text.

        LIMITATION (CJK): the substring fallback has no morpheme boundary, so a
        short expected entity can match inside a larger word (e.g. expected
        "가나" matches within "가나상사"), which can *over-count* recall in the
        multilingual case. We accept this rather than apply ASCII-style boundary
        checks, because CJK attaches particles/suffixes directly to a word
        (Korean "가나가", Japanese "トヨタは"), so a boundary check would instead
        *under-count* legitimate mentions. Faithful matching here needs a
        morphological segmenter (e.g. nori); recall is the only emitted metric,
        so the bias is conservative-to-optimistic, not a correctness gate.
        """
        if cls._is_spaceless_script(phrase):
            cleaned = phrase.strip().lower()
            return bool(cleaned) and cleaned in answer_text.lower()

        phrase_tokens = cls._tokenize(phrase)
        if not phrase_tokens:
            return False
        window = len(phrase_tokens)
        for start in range(len(answer_tokens) - window + 1):
            if answer_tokens[start : start + window] == phrase_tokens:
                return True
        return False

    @classmethod
    def _coverage(cls, expected: list[str], answer: str) -> tuple[float | None, int]:
        """Return (coverage, num_matched) for expected-artifacts-in-answer.

        Coverage = matched / expected (i.e. recall): the fraction of expected
        graph artifacts whose full word sequence appears, word-boundary aware,
        in the answer (so "AI" does not match inside "airport"). We intentionally
        report ONLY coverage — not precision/F1 — because precision would require
        enumerating the answer's own entities/relationships, which we cannot do
        from free text; emitting precision as a copy of recall (the previous
        behaviour) overstated the signal.

        Returns ``None`` when nothing is expected for the dimension, so a query
        with (say) only expected entities is not penalized for having no expected
        relationships when the overall score is averaged.
        """
        if not expected:
            return None, 0
        answer_tokens = cls._tokenize(answer)
        matched = sum(
            1
            for item in expected
            if item and cls._phrase_in_tokens(item, answer_tokens, answer)
        )
        return matched / len(expected), matched

    def _build_metrics(
        self, result: EvaluationResult
    ) -> tuple[list[EvaluationMetric], dict[str, Any]]:
        answer = result.generated_answer or ""
        expected_entities = result.metadata.get("expected_entities", []) or []
        expected_relationships = result.metadata.get("expected_relationships", []) or []

        e_cov, e_matched = self._coverage(expected_entities, answer)
        r_cov, r_matched = self._coverage(expected_relationships, answer)

        # Only emit metrics for dimensions that actually have expectations, so a
        # missing dimension does not dilute the averaged overall score.
        metrics: list[EvaluationMetric] = []
        if expected_entities and e_cov is not None:
            metrics.append(
                EvaluationMetric(
                    metric_type=EvaluationMetricType.ENTITY_COVERAGE, value=e_cov
                )
            )
        if expected_relationships and r_cov is not None:
            metrics.append(
                EvaluationMetric(
                    metric_type=EvaluationMetricType.RELATIONSHIP_COVERAGE, value=r_cov
                )
            )
        metadata = {
            **self._extract_search_metadata(result),
            "expected_entity_count": len(expected_entities),
            "matched_entity_count": e_matched,
            "expected_relationship_count": len(expected_relationships),
            "matched_relationship_count": r_matched,
        }
        return metrics, metadata

    def evaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        metrics, metadata = self._build_metrics(result)
        scored = [m.value for m in metrics if m.value is not None]
        overall = sum(scored) / len(scored) if scored else 0.0
        return EvaluationReport(
            query_id=query.query_id,
            evaluator_type=self.evaluator_type,
            metrics=metrics,
            overall_score=overall,
            evaluation_time=datetime.now(),
            metadata=metadata,
        )

    async def aevaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        # Deterministic and CPU-only; reuse the sync path.
        return self.evaluate_single(query, result, ground_truth, **kwargs)
