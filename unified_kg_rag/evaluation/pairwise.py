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

LLM judges also prefer the longer answer (Zeng et al., arXiv 2506.06331).
Every comparison therefore reports, under ``length``, each judged pair's
answer lengths in estimated tokens (``estimate_token_count``: script-aware,
so Korean, Japanese or Chinese answers are not measured in whitespace
words) and the candidate's win rate per bucket of the length ratio
(candidate over baseline), with the correlation between the relative length
difference and the pair's score per criterion. With
``evaluation.pairwise_mode: differential`` a fast model
(``models.roles.pairwise_overlap``) first splits the two answers into the
statements both make and those only one makes; the judge then decides each
criterion on the unique statements, the shared ones shown as common ground
that earns neither side credit (both orders, ties and position consistency
as in the standard mode). A pair with no unique statement is a tie without a
judge call (``identical_pairs``); a failed or malformed split fails the pair.
docs/design.md#92-length-controlled-judging-assertions-and-the-closed-book-arm
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from unified_kg_rag.adapters.evaluators.base import judge_model_kwargs
from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.llm.langchain import LLMJsonOutputParser
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
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
    PairwiseDifferentialOverallPrompt,
    PairwiseDifferentialPrompt,
    PairwiseOverlapPrompt,
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
# The keys of the differential mode's statement split.
_OVERLAP_KEYS = ("shared", "unique_1", "unique_2")
# Buckets of the answer-length ratio (candidate over baseline, in estimated
# tokens), shortest first. 0.9-1.1 (both ends included) holds the pairs whose
# lengths differ by about a tenth or less; 1.5 is half again as long.
LENGTH_BUCKETS = ("<0.67x", "0.67-0.9x", "0.9-1.1x", "1.1-1.5x", ">1.5x")
# A correlation over fewer pairs says nothing.
_MIN_CORRELATION_PAIRS = 3

# Per judged pair: the outcome per criterion, whether the two orders agreed
# per criterion (None: no judge call, a differential pair with nothing
# unique), and the differential split (None in the standard mode).
_Judged = tuple[dict[str, str], dict[str, bool] | None, dict[str, list[str]] | None]


def length_bucket(ratio: float) -> str:
    """The ``LENGTH_BUCKETS`` label of a candidate/baseline length ratio."""
    if ratio < 2 / 3:
        return LENGTH_BUCKETS[0]
    if ratio < 0.9:
        return LENGTH_BUCKETS[1]
    if ratio <= 1.1:
        return LENGTH_BUCKETS[2]
    if ratio < 1.5:
        return LENGTH_BUCKETS[3]
    return LENGTH_BUCKETS[4]


def answer_lengths(baseline: str, candidate: str) -> dict[str, Any]:
    """Both answers' estimated token counts, their ratio (candidate over
    baseline) and relative difference ((candidate - baseline) over the
    longer one, in [-1, 1])."""
    base = estimate_token_count(baseline)
    cand = estimate_token_count(candidate)
    longer = max(base, cand)
    ratio = cand / base if base else math.inf
    return {
        "baseline_tokens": base,
        "candidate_tokens": cand,
        "ratio": ratio,
        "relative_difference": (cand - base) / longer if longer else 0.0,
        "bucket": length_bucket(ratio),
    }


def length_summary(
    lengths: Mapping[str, Mapping[str, Any]],
    outcomes: Mapping[str, Mapping[str, str]],
    criteria: Sequence[str],
) -> dict[str, Any]:
    """Win rates per length bucket and the length-score correlation per
    criterion, over the pairs of ``outcomes`` that ``lengths`` measures."""
    pairs = [q for q in outcomes if q in lengths]
    buckets: dict[str, Any] = {}
    for label in LENGTH_BUCKETS:
        in_bucket = [q for q in pairs if lengths[q]["bucket"] == label]
        per_criterion: dict[str, Any] = {}
        for criterion in criteria:
            counts = Counter(outcomes[q][criterion] for q in in_bucket)
            scores = [_SCORES[outcomes[q][criterion]] for q in in_bucket]
            per_criterion[criterion] = {
                "candidate_wins": counts["candidate"],
                "baseline_wins": counts["baseline"],
                "ties": counts["tie"],
                "candidate_win_rate": (
                    float(sum(scores) / len(scores)) if scores else None
                ),
            }
        buckets[label] = {"pairs": len(in_bucket), "criteria": per_criterion}
    correlation: dict[str, float | None] = {}
    differences = np.asarray(
        [lengths[q]["relative_difference"] for q in pairs], dtype=float
    )
    for criterion in criteria:
        scores_array = np.asarray(
            [_SCORES[outcomes[q][criterion]] for q in pairs], dtype=float
        )
        correlation[criterion] = (
            float(np.corrcoef(differences, scores_array)[0, 1])
            if len(pairs) >= _MIN_CORRELATION_PAIRS
            and differences.std() > 0
            and scores_array.std() > 0
            else None
        )
    candidate = [lengths[q]["candidate_tokens"] for q in pairs]
    baseline = [lengths[q]["baseline_tokens"] for q in pairs]
    return {
        "unit": "estimated_tokens",
        "pairs": len(pairs),
        "mean_baseline_tokens": float(np.mean(baseline)) if pairs else None,
        "mean_candidate_tokens": float(np.mean(candidate)) if pairs else None,
        "buckets": buckets,
        "score_correlation": correlation,
        "per_query": {q: dict(lengths[q]) for q in sorted(pairs)},
    }


def _statement_list(statements: Sequence[str]) -> str:
    return "\n".join(f"- {s}" for s in statements) if statements else "(none)"


class PairwiseJudge:
    """Judges a candidate strategy's answers against the baseline's."""

    def __init__(self, config: Config, providers: Providers | None = None) -> None:
        self.config = config
        providers = Providers.resolve(config, providers)
        self.mode = config.evaluation.pairwise_mode
        overall = config.evaluation.pairwise_overall
        prompt: type[PairwiseComparisonPrompt] | type[PairwiseDifferentialPrompt]
        if self.mode == "differential":
            prompt = (
                PairwiseDifferentialOverallPrompt
                if overall
                else PairwiseDifferentialPrompt
            )
        else:
            prompt = (
                PairwiseComparisonOverallPrompt if overall else PairwiseComparisonPrompt
            )
        self.criteria: tuple[str, ...] = tuple(prompt.output_variables or ())
        self.chain = setup_chain(
            providers.llm_factory,
            config.model_id(ModelRole.EVALUATION),
            prompt,
            LLMJsonOutputParser(answer_keys=self.criteria),
            custom_prompts=config.custom_prompts,
            model_purpose=ModelPurpose.EVALUATION,
            **judge_model_kwargs(config),
        )
        self.overlap_chain = (
            setup_chain(
                providers.llm_factory,
                config.model_id(ModelRole.PAIRWISE_OVERLAP),
                PairwiseOverlapPrompt,
                LLMJsonOutputParser(answer_keys=_OVERLAP_KEYS),
                custom_prompts=config.custom_prompts,
                model_purpose=ModelPurpose.EVALUATION,
                **judge_model_kwargs(config),
            )
            if self.mode == "differential"
            else None
        )
        self._semaphore = asyncio.Semaphore(config.processing.max_concurrency)

    async def _verdicts(self, query: str, first: str, second: str) -> dict[str, str]:
        return await self._ask({"query": query, "answer1": first, "answer2": second})

    async def _ask(self, inputs: dict[str, str]) -> dict[str, str]:
        """The judge's verdict per criterion for one order of the inputs."""
        async with self._semaphore:
            output = await self.chain.ainvoke(inputs)
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

    async def _ajudge_both_orders(
        self, query: str, baseline: str, candidate: str
    ) -> tuple[dict[str, str], dict[str, bool]]:
        """The outcome per criterion (``"candidate"``, ``"baseline"`` or
        ``"tie"``), and whether the two orders agreed.

        Judged as (baseline, candidate) and (candidate, baseline); a verdict
        that flips with the order is a tie.
        """
        forward, backward = await asyncio.gather(
            self._verdicts(query, baseline, candidate),
            self._verdicts(query, candidate, baseline),
        )
        return self._combine(forward, backward)

    def _combine(
        self, forward: dict[str, str], backward: dict[str, str]
    ) -> tuple[dict[str, str], dict[str, bool]]:
        """The outcome per criterion of the (baseline, candidate) verdicts
        ``forward`` and the (candidate, baseline) ones ``backward``."""
        outcome: dict[str, str] = {}
        consistent: dict[str, bool] = {}
        for criterion in self.criteria:
            one = {"1": "baseline", "2": "candidate"}.get(forward[criterion], "tie")
            two = {"1": "candidate", "2": "baseline"}.get(backward[criterion], "tie")
            consistent[criterion] = one == two
            outcome[criterion] = one if one == two else "tie"
        return outcome, consistent

    async def _split(
        self, query: str, baseline: str, candidate: str
    ) -> dict[str, list[str]]:
        """The differential mode's statement split: ``shared``, ``baseline``
        (only the baseline's) and ``candidate`` (only the candidate's).

        Raises:
            EvaluationException: The reply lacks a key or a key is not a list
                of strings.
        """
        if self.overlap_chain is None:
            raise EvaluationException("the differential mode has no overlap model")
        async with self._semaphore:
            output = await self.overlap_chain.ainvoke(
                {"query": query, "answer1": baseline, "answer2": candidate}
            )
        split: dict[str, list[str]] = {}
        for key, side in zip(
            _OVERLAP_KEYS, ("shared", "baseline", "candidate"), strict=True
        ):
            value = output.get(key) if isinstance(output, dict) else None
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                # A missing or malformed list is a failed pair, not "nothing".
                raise EvaluationException(
                    f"pairwise overlap gave no valid '{key}' list: {value!r}"
                )
            split[side] = [item.strip() for item in value if item.strip()]
        return split

    async def _ajudge_differential(
        self, query: str, baseline: str, candidate: str
    ) -> _Judged:
        """Split the answers, then judge the unique statements in both orders;
        a pair with no unique statement ties without a judge call."""
        split = await self._split(query, baseline, candidate)
        if not split["baseline"] and not split["candidate"]:
            return dict.fromkeys(self.criteria, "tie"), None, split
        shared = _statement_list(split["shared"])
        base, cand = _statement_list(split["baseline"]), _statement_list(
            split["candidate"]
        )
        forward, backward = await asyncio.gather(
            self._ask(
                {"query": query, "shared": shared, "unique1": base, "unique2": cand}
            ),
            self._ask(
                {"query": query, "shared": shared, "unique1": cand, "unique2": base}
            ),
        )
        outcome, consistent = self._combine(forward, backward)
        return outcome, consistent, split

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
        judged_ids: list[str] = []
        lengths: dict[str, dict[str, Any]] = {}
        splits: dict[str, dict[str, Any]] = {}
        for (base, cand), verdict in zip(to_judge, judged, strict=True):
            if verdict is None:
                continue
            outcome, agreed, split = verdict
            outcomes[cand.query_id] = outcome
            judged_ids.append(cand.query_id)
            if agreed is not None:
                agreements.append(agreed)
            lengths[cand.query_id] = answer_lengths(
                base.generated_answer, cand.generated_answer
            )
            if split is not None:
                lengths[cand.query_id].update(
                    {
                        f"{side}_unique_tokens": estimate_token_count(
                            "\n".join(split[side])
                        )
                        for side in ("baseline", "candidate")
                    }
                )
                lengths[cand.query_id]["shared_tokens"] = estimate_token_count(
                    "\n".join(split["shared"])
                )
                splits[cand.query_id] = split
        decided = list(outcomes.values())
        criteria: dict[str, Any] = {}
        for criterion in self.criteria:
            counts = Counter(o[criterion] for o in decided)
            scores = [_SCORES[o[criterion]] for o in decided]
            # Ties split evenly, as the papers report win rates.
            rate, low, high = None, None, None
            if judged_ids:
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
        result: dict[str, Any] = {
            "mode": self.mode,
            "pairs": len(pairs),
            "judged": len(judged_ids),
            "failed": sum(v is None for v in judged),
            "candidate_errors": errors["candidate_errors"],
            "baseline_errors": errors["baseline_errors"],
            "both_errors": errors["both_errors"],
            "unanswerable_skipped": errors["unanswerable_skipped"],
            "position_consistency": consistency,
            "bootstrap": dict(_BOOTSTRAP),
            "criteria": criteria,
            "per_query": dict(sorted(outcomes.items())),
            # Judged pairs only: a pair decided by a failed answer has no
            # length to compare.
            "length": length_summary(
                lengths, {q: outcomes[q] for q in judged_ids}, self.criteria
            ),
        }
        if self.mode == "differential":
            result["identical_pairs"] = len(judged_ids) - len(agreements)
            result["differential"] = {"per_query": dict(sorted(splits.items()))}
        return result

    async def _safe_judge(
        self, baseline: EvaluationResult, candidate: EvaluationResult
    ) -> _Judged | None:
        try:
            if self.mode == "differential":
                return await self._ajudge_differential(
                    candidate.question,
                    baseline.generated_answer,
                    candidate.generated_answer,
                )
            outcome, consistent = await self._ajudge_both_orders(
                candidate.question,
                baseline.generated_answer,
                candidate.generated_answer,
            )
            return outcome, consistent, None
        except Exception as e:  # noqa: BLE001 - a failed pair is counted, not fatal
            logger.warning("Pairwise judge failed for '%s': %s", candidate.query_id, e)
            return None


def _answered(result: EvaluationResult) -> bool:
    return bool(result.generated_answer) and not result.error


def _unanswerable(result: EvaluationResult) -> bool:
    return result.metadata.get("answerable") is False
