# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pairwise LLM judge: which of two strategies' answers is better, per criterion.

The GraphRAG and LightRAG papers compare methods head to head with an LLM
judge on comprehensiveness, diversity, empowerment and directness rather
than against a gold answer, because global sensemaking questions have none.
``PairwiseJudge`` does the same for ``run-eval --pairwise-judge`` over two
strategies' answers to the same queries. LLM judges prefer the answer they
read first (position bias), so each pair is judged in both orders and a
criterion counts as a win only when both orders pick the same answer;
otherwise it is a tie. ``position_consistency`` is the share of judged pairs
whose two orders agreed: a judge near 0 decides by position, and its "ties"
say nothing about the answers. A missing or malformed verdict fails the pair
(counted in ``failed``) rather than counting as a tie. A pair where one side's
answer generation failed is not judged: the side that answered wins every
criterion (tallied as ``candidate_errors`` / ``baseline_errors``), so a
strategy cannot improve its win rate by failing on hard queries. A pair both
sides failed counts in ``both_errors`` only, and an item marked
``metadata.answerable: false`` is not judged (``unanswerable_skipped``): it
has no answer to prefer.

The win rate scores each decided pair 1 (candidate), 0.5 (tie) or 0
(baseline) and comes with a bootstrap interval over those per-query scores
(``comparison.bootstrap_ci``: 2,000 resamples, 95%, seed 0). It is ``None``
when the judge decided no pair, since the pairs decided by a failed answer
alone say nothing about the judge's preference. With
``evaluation.pairwise_overall`` the judge also names an ``overall`` winner,
as LightRAG's prompt does.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from unified_kg_rag.adapters.evaluators.base import judge_model_kwargs
from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.llm.langchain import LLMJsonOutputParser
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import (
    Config,
    EvaluationResult,
    ModelPurpose,
    ModelRole,
)
from unified_kg_rag.domain.prompts import (
    PairwiseComparisonOverallPrompt,
    PairwiseComparisonPrompt,
)
from unified_kg_rag.shared import EvaluationException, get_logger, observability

from .comparison import bootstrap_ci

logger = get_logger(__name__)

# The default criteria (evaluation.pairwise_overall adds "overall").
CRITERIA = tuple(PairwiseComparisonPrompt.output_variables or ())
_VERDICTS = frozenset({"1", "2", "tie"})
# A decided pair's score for the candidate, per outcome.
_SCORES = {"candidate": 1.0, "tie": 0.5, "baseline": 0.0}
_BOOTSTRAP: dict[str, Any] = {"resamples": 2000, "confidence": 0.95, "seed": 0}


class PairwiseJudge:
    """Judges a candidate strategy's answers against the baseline's."""

    def __init__(self, config: Config, providers: Providers | None = None) -> None:
        self.config = config
        providers = Providers.resolve(config, providers)
        prompt = (
            PairwiseComparisonOverallPrompt
            if config.evaluation.pairwise_overall
            else PairwiseComparisonPrompt
        )
        self.criteria: tuple[str, ...] = tuple(prompt.output_variables or ())
        self.chain = setup_chain(
            providers.llm_factory,
            config.model_id(ModelRole.EVALUATION),
            prompt,
            LLMJsonOutputParser(),
            custom_prompts=config.custom_prompts,
            model_purpose=ModelPurpose.EVALUATION,
            **judge_model_kwargs(config),
        )
        self._semaphore = asyncio.Semaphore(config.processing.max_concurrency)

    async def _verdicts(self, query: str, first: str, second: str) -> dict[str, str]:
        async with self._semaphore:
            output = await self.chain.ainvoke(
                {"query": query, "answer1": first, "answer2": second}
            )
        verdicts: dict[str, str] = {}
        for criterion in self.criteria:
            entry = output.get(criterion) if isinstance(output, dict) else None
            winner = entry.get("winner") if isinstance(entry, dict) else entry
            verdict = str(winner).strip().strip('"').lower()
            if verdict not in _VERDICTS:
                # A missing or malformed verdict is a failed judgment, not a tie.
                raise EvaluationException(
                    f"pairwise judge gave no valid '{criterion}' winner: {winner!r}"
                )
            verdicts[criterion] = verdict
        return verdicts

    async def ajudge(self, query: str, baseline: str, candidate: str) -> dict[str, str]:
        """Per criterion: ``"candidate"``, ``"baseline"`` or ``"tie"``.

        Judged as (baseline, candidate) and (candidate, baseline); a verdict
        that flips with the order is a tie.
        """
        outcome, _ = await self._ajudge_both_orders(query, baseline, candidate)
        return outcome

    async def _ajudge_both_orders(
        self, query: str, baseline: str, candidate: str
    ) -> tuple[dict[str, str], dict[str, bool]]:
        """The outcome per criterion, and whether the two orders agreed."""
        forward, backward = await asyncio.gather(
            self._verdicts(query, baseline, candidate),
            self._verdicts(query, candidate, baseline),
        )
        outcome: dict[str, str] = {}
        consistent: dict[str, bool] = {}
        for criterion in self.criteria:
            one = {"1": "baseline", "2": "candidate"}.get(forward[criterion], "tie")
            two = {"1": "candidate", "2": "baseline"}.get(backward[criterion], "tie")
            consistent[criterion] = one == two
            outcome[criterion] = one if one == two else "tie"
        return outcome, consistent

    async def acompare(
        self, baseline: list[EvaluationResult], candidate: list[EvaluationResult]
    ) -> dict[str, Any]:
        """Win rates of ``candidate`` over ``baseline``, per criterion.

        ``failed`` counts the pairs the judge could not decide (its calls
        failed); ``run-eval`` holds it to ``--max-failure-rate``.
        """
        by_id = {r.query_id: r for r in baseline}
        pairs = [(by_id[r.query_id], r) for r in candidate if r.query_id in by_id]
        outcomes: dict[str, dict[str, str]] = {}
        errors: Counter[str] = Counter()
        to_judge: list[tuple[EvaluationResult, EvaluationResult]] = []
        for base, cand in pairs:
            if _unanswerable(base) or _unanswerable(cand):
                errors["unanswerable_skipped"] += 1
                continue
            base_ok, cand_ok = _answered(base), _answered(cand)
            if base_ok and cand_ok:
                to_judge.append((base, cand))
            elif base_ok:
                errors["candidate_errors"] += 1
                outcomes[cand.query_id] = dict.fromkeys(self.criteria, "baseline")
            elif cand_ok:
                errors["baseline_errors"] += 1
                outcomes[cand.query_id] = dict.fromkeys(self.criteria, "candidate")
            else:
                errors["both_errors"] += 1
        # The judge calls nest under one span instead of each starting a root
        # trace, as the other evaluators' do.
        with observability.span(
            "kg_rag.evaluate", evaluator="pairwise_judge", items=len(to_judge)
        ):
            judged = await asyncio.gather(
                *(self._safe_judge(b, c) for b, c in to_judge)
            )
        agreements: list[dict[str, bool]] = []
        for (_, cand), verdict in zip(to_judge, judged, strict=True):
            if verdict is not None:
                outcomes[cand.query_id], agreed = verdict
                agreements.append(agreed)
        decided = list(outcomes.values())
        criteria: dict[str, Any] = {}
        for criterion in self.criteria:
            counts = Counter(o[criterion] for o in decided)
            scores = [_SCORES[o[criterion]] for o in decided]
            # Ties split evenly, as the papers report win rates.
            rate, low, high = None, None, None
            if agreements:
                rate = float(sum(scores) / len(scores))
                low, high = bootstrap_ci(scores, **_BOOTSTRAP)
            criteria[criterion] = {
                "candidate_wins": counts["candidate"],
                "baseline_wins": counts["baseline"],
                "ties": counts["tie"],
                "candidate_win_rate": rate,
                "ci_low": low,
                "ci_high": high,
                "decided": counts["candidate"] + counts["baseline"],
                "position_consistency": (
                    sum(a[criterion] for a in agreements) / len(agreements)
                    if agreements
                    else None
                ),
            }
        consistency = (
            sum(sum(a.values()) for a in agreements)
            / (len(agreements) * len(self.criteria))
            if agreements
            else None
        )
        if consistency is not None and consistency < 0.5:
            logger.warning(
                "Pairwise judge: the two answer orders agreed on %.0f%% of the "
                "verdicts; most ties are position bias, not equal answers.",
                consistency * 100,
            )
        return {
            "pairs": len(pairs),
            "judged": len(agreements),
            "failed": sum(v is None for v in judged),
            "candidate_errors": errors["candidate_errors"],
            "baseline_errors": errors["baseline_errors"],
            "both_errors": errors["both_errors"],
            "unanswerable_skipped": errors["unanswerable_skipped"],
            "position_consistency": consistency,
            "bootstrap": dict(_BOOTSTRAP),
            "criteria": criteria,
            "per_query": dict(sorted(outcomes.items())),
        }

    async def _safe_judge(
        self, baseline: EvaluationResult, candidate: EvaluationResult
    ) -> tuple[dict[str, str], dict[str, bool]] | None:
        try:
            return await self._ajudge_both_orders(
                candidate.question,
                baseline.generated_answer,
                candidate.generated_answer,
            )
        except Exception as e:  # noqa: BLE001 - a failed pair is counted, not fatal
            logger.warning("Pairwise judge failed for '%s': %s", candidate.query_id, e)
            return None


def _answered(result: EvaluationResult) -> bool:
    return bool(result.generated_answer) and not result.error


def _unanswerable(result: EvaluationResult) -> bool:
    return result.metadata.get("answerable") is False
