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
otherwise it is a tie. A pair where one side's answer generation failed is
not judged: the side that answered wins every criterion (tallied as
``candidate_errors`` / ``baseline_errors``), so a strategy cannot improve its
win rate by failing on hard queries. A pair both sides failed counts in
``both_errors`` only.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from typing import Any

from langchain_core.output_parsers import JsonOutputParser

from unified_kg_rag.adapters.evaluators.base import judge_model_kwargs
from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import (
    Config,
    EvaluationResult,
    ModelPurpose,
    ModelRole,
)
from unified_kg_rag.domain.prompts import PairwiseComparisonPrompt
from unified_kg_rag.shared import get_logger, observability

logger = get_logger(__name__)

CRITERIA = tuple(PairwiseComparisonPrompt.output_variables or ())


class PairwiseJudge:
    """Judges a candidate strategy's answers against the baseline's."""

    def __init__(self, config: Config, providers: Providers | None = None) -> None:
        self.config = config
        providers = Providers.resolve(config, providers)
        self.chain = setup_chain(
            providers.llm_factory,
            config.model_id(ModelRole.EVALUATION),
            PairwiseComparisonPrompt,
            JsonOutputParser(),
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
        for criterion in CRITERIA:
            entry = output.get(criterion) if isinstance(output, dict) else None
            winner = entry.get("winner") if isinstance(entry, dict) else entry
            verdicts[criterion] = str(winner).strip().strip('"').lower()
        return verdicts

    async def ajudge(self, query: str, baseline: str, candidate: str) -> dict[str, str]:
        """Per criterion: ``"candidate"``, ``"baseline"`` or ``"tie"``.

        Judged as (baseline, candidate) and (candidate, baseline); a verdict
        that flips with the order is a tie.
        """
        forward, backward = await asyncio.gather(
            self._verdicts(query, baseline, candidate),
            self._verdicts(query, candidate, baseline),
        )
        outcome: dict[str, str] = {}
        for criterion in CRITERIA:
            one = {"1": "baseline", "2": "candidate"}.get(forward[criterion], "tie")
            two = {"1": "candidate", "2": "baseline"}.get(backward[criterion], "tie")
            outcome[criterion] = one if one == two else "tie"
        return outcome

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
            base_ok, cand_ok = _answered(base), _answered(cand)
            if base_ok and cand_ok:
                to_judge.append((base, cand))
            elif base_ok:
                errors["candidate_errors"] += 1
                outcomes[cand.query_id] = dict.fromkeys(CRITERIA, "baseline")
            elif cand_ok:
                errors["baseline_errors"] += 1
                outcomes[cand.query_id] = dict.fromkeys(CRITERIA, "candidate")
            else:
                errors["both_errors"] += 1
        # The judge calls nest under one span instead of each starting a root
        # trace, as the other evaluators' do.
        with observability.span(
            "kg_rag.evaluate", evaluator="pairwise_judge", items=len(to_judge)
        ):
            judged: list[dict[str, str] | None] = await asyncio.gather(
                *(self._safe_judge(b, c) for b, c in to_judge)
            )
        for (_, cand), outcome in zip(to_judge, judged, strict=True):
            if outcome is not None:
                outcomes[cand.query_id] = outcome
        decided = list(outcomes.values())
        criteria: dict[str, Any] = {}
        for criterion in CRITERIA:
            counts = Counter(o[criterion] for o in decided)
            criteria[criterion] = {
                "candidate_wins": counts["candidate"],
                "baseline_wins": counts["baseline"],
                "ties": counts["tie"],
                # Ties split evenly, as the papers report win rates.
                "candidate_win_rate": (
                    (counts["candidate"] + counts["tie"] / 2) / len(decided)
                    if decided
                    else None
                ),
                "decided": counts["candidate"] + counts["baseline"],
            }
        return {
            "pairs": len(pairs),
            "judged": sum(o is not None for o in judged),
            "failed": sum(o is None for o in judged),
            "candidate_errors": errors["candidate_errors"],
            "baseline_errors": errors["baseline_errors"],
            "both_errors": errors["both_errors"],
            "criteria": criteria,
            "per_query": dict(sorted(outcomes.items())),
        }

    async def _safe_judge(
        self, baseline: EvaluationResult, candidate: EvaluationResult
    ) -> dict[str, str] | None:
        try:
            return await self.ajudge(
                candidate.question,
                baseline.generated_answer,
                candidate.generated_answer,
            )
        except Exception as e:  # noqa: BLE001 - a failed pair is counted, not fatal
            logger.warning("Pairwise judge failed for '%s': %s", candidate.query_id, e)
            return None


def _answered(result: EvaluationResult) -> bool:
    return bool(result.generated_answer) and not result.error
