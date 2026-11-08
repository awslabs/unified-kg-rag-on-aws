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
as in the standard mode). The split itself reads the answers in a slot
order, and a split model favours one slot (x5, 2026-11-08: an A/A pair won
0.645 with the candidate always Answer 2, 0.52 swapped), so every pair is
split in both slot orders (``SPLIT_SLOTS``), each split judged in both
orders, and the pair's score per criterion is the mean of the two slots'
(its outcome the side that mean favours); ``slots`` reports the candidate's
win rate per slot and their difference (``slot_effect``). A slot with no
unique statement is a tie without a judge call, a pair with none in either
slot counts in ``identical_pairs``; a failed or malformed split fails the
pair.

Each criterion also has a sign-flip ``p_value`` (``comparison.
sign_flip_p_value`` over the per-query scores minus 0.5, unadjusted) and its
``per_query`` scores; ``comparison.adjust_for_multiplicity`` adds the Holm
value over the comparison's pairwise family.
docs/design.md#93-length-controlled-judging-assertions-and-the-closed-book-arm
"""

from __future__ import annotations

import asyncio
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from unified_kg_rag.adapters.evaluators.base import (
    error_digest,
    judge_answers,
    judge_effort,
    judge_model_kwargs,
)
from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import (
    Config,
    EvaluationResult,
    ModelPurpose,
    ModelRole,
)
from unified_kg_rag.domain.models.config import EFFORT_THINKING_TOKENS
from unified_kg_rag.domain.prompts import (
    PairwiseComparisonOverallPrompt,
    PairwiseComparisonPrompt,
    PairwiseDifferentialOverallPrompt,
    PairwiseDifferentialPrompt,
    PairwiseOverlapPrompt,
)
from unified_kg_rag.domain.retrieval.citations import strip_markers
from unified_kg_rag.shared import EvaluationException, get_logger, observability

from .comparison import bootstrap_ci, sign_flip_p_value

logger = get_logger(__name__)

# The default criteria (evaluation.pairwise_overall adds "overall").
CRITERIA = tuple(PairwiseComparisonPrompt.output_variables or ())
_VERDICTS = frozenset({"1", "2", "tie"})
# A decided pair's score for the candidate, per outcome.
_SCORES = {"candidate": 1.0, "tie": 0.5, "baseline": 0.0}
_BOOTSTRAP: dict[str, Any] = {"resamples": 2000, "confidence": 0.95, "seed": 0}
# The keys of the differential mode's statement split.
_OVERLAP_KEYS = ("shared", "unique_1", "unique_2")
# The differential mode's split slot orders: which answer the split reads
# as Answer 1. Every pair is split in both.
SPLIT_SLOTS = ("baseline_first", "candidate_first")


def overlap_output_floor(config: Config) -> int | None:
    """The overlap split's output-token floor: the default cap
    (``models.default_max_output_tokens``) above the thinking budget of
    the effort the split runs at (``judge_effort``), since it copies both
    answers' statements after it reasons and thinking counts toward
    max_tokens; ``None`` (the prompt's own floor) when the cap is the
    model maximum."""
    cap = config.models.default_max_output_tokens
    if cap is None:
        return None
    effort = judge_effort(config, config.model_id(ModelRole.PAIRWISE_OVERLAP))
    return cap + EFFORT_THINKING_TOKENS.get(effort, 0)


# Buckets of the answer-length ratio (candidate over baseline, in estimated
# tokens), shortest first, symmetric in the log ratio: a candidate 1.5 times
# as long as the baseline and one 1/1.5 (0.67) times as long sit in mirror
# buckets. The middle one holds the pairs within a factor 1.1 either way
# (0.91-1.1, both ends included); the outer ones a factor 1.5 or more.
LENGTH_BUCKETS = ("<=0.67x", "0.67-0.91x", "0.91-1.1x", "1.1-1.5x", ">=1.5x")
_NEAR, _FAR = math.log(1.1), math.log(1.5)
# Float error on a ratio such as 2/3 must not move it off its boundary.
_LOG_TOLERANCE = 1e-9
# A correlation over fewer pairs says nothing.
_MIN_CORRELATION_PAIRS = 3

# One judged slot of a pair: the outcome per criterion and whether the two
# judge orders agreed per criterion (None: no judge call, a differential
# split with nothing unique).
_Slot = tuple[dict[str, str], dict[str, bool] | None]
# Per judged pair: its slots (one in the standard mode, ``SPLIT_SLOTS`` in
# the differential mode) and the differential splits by slot (None in the
# standard mode).
_Judged = tuple[list[_Slot], dict[str, dict[str, list[str]]] | None]


def length_bucket(ratio: float) -> str:
    """The ``LENGTH_BUCKETS`` label of a candidate/baseline length ratio, by
    ``|log ratio|``: a ratio and its reciprocal land in mirror buckets."""
    if ratio <= 0:
        return LENGTH_BUCKETS[0]
    if math.isinf(ratio):
        return LENGTH_BUCKETS[4]
    spread = abs(math.log(ratio))
    if spread <= _NEAR + _LOG_TOLERANCE:
        return LENGTH_BUCKETS[2]
    if spread < _FAR - _LOG_TOLERANCE:
        return LENGTH_BUCKETS[1] if ratio < 1 else LENGTH_BUCKETS[3]
    return LENGTH_BUCKETS[0] if ratio < 1 else LENGTH_BUCKETS[4]


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
            judge_answers(providers.llm_factory, config, self.criteria),
            custom_prompts=config.custom_prompts,
            model_purpose=ModelPurpose.EVALUATION,
            **judge_model_kwargs(config),
        )
        self.overlap_chain = (
            setup_chain(
                providers.llm_factory,
                config.model_id(ModelRole.PAIRWISE_OVERLAP),
                PairwiseOverlapPrompt,
                judge_answers(providers.llm_factory, config, _OVERLAP_KEYS),
                custom_prompts=config.custom_prompts,
                model_purpose=ModelPurpose.EVALUATION,
                min_output_tokens=overlap_output_floor(config),
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
        self, query: str, baseline: str, candidate: str, slot: str
    ) -> dict[str, list[str]]:
        """The differential mode's statement split: ``shared``, ``baseline``
        (only the baseline's) and ``candidate`` (only the candidate's), with
        the answers in ``slot``'s order (``SPLIT_SLOTS``).

        Raises:
            EvaluationException: The reply lacks a key or a key is not a list
                of strings.
        """
        if self.overlap_chain is None:
            raise EvaluationException("the differential mode has no overlap model")
        swapped = slot == "candidate_first"
        first, second = (candidate, baseline) if swapped else (baseline, candidate)
        async with self._semaphore:
            output = await self.overlap_chain.ainvoke(
                {"query": query, "answer1": first, "answer2": second}
            )
        sides = (
            ("shared", "candidate", "baseline")
            if swapped
            else (
                "shared",
                "baseline",
                "candidate",
            )
        )
        split: dict[str, list[str]] = {}
        for key, side in zip(_OVERLAP_KEYS, sides, strict=True):
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
        """Split the answers in both slot orders, then judge each split's
        unique statements in both orders; a split with no unique statement
        ties without a judge call."""
        splits = await asyncio.gather(
            *(self._split(query, baseline, candidate, slot) for slot in SPLIT_SLOTS)
        )
        slots = await asyncio.gather(
            *(self._judge_split(query, split) for split in splits)
        )
        return list(slots), dict(zip(SPLIT_SLOTS, splits, strict=True))

    async def _judge_split(self, query: str, split: dict[str, list[str]]) -> _Slot:
        """One split's outcome: its unique statements judged in both orders."""
        if not split["baseline"] and not split["candidate"]:
            return dict.fromkeys(self.criteria, "tie"), None
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
        return self._combine(forward, backward)

    async def acompare(
        self, baseline: list[EvaluationResult], candidate: list[EvaluationResult]
    ) -> dict[str, Any]:
        """Win rates of ``candidate`` over ``baseline``, per criterion.

        ``failed`` counts the pairs the judge could not decide (its calls
        failed); ``run-eval`` holds it to ``--max-failure-rate``.
        """
        return await self.acompare_repeats([baseline], [candidate])

    async def acompare_repeats(
        self,
        baselines: Sequence[list[EvaluationResult]],
        candidates: Sequence[list[EvaluationResult]],
    ) -> dict[str, Any]:
        """``acompare`` over repeats: repeat r of the candidate against
        repeat r of the baseline, for as many repeats as both have
        (``repeats``). With one repeat it is ``acompare``.

        The pairs of one query are clustered: the win rate averages each
        query's scores over its decided repeats first, then over the
        queries, and its interval resamples queries (a cluster bootstrap),
        so judging three repeats is not read as three times the queries.
        The counts (wins, ties, ``judged``, ``failed``, the error counts)
        are over every pair; ``per_query`` lists each query's outcomes in
        repeat order (``None`` where that repeat's pair was not decided),
        and the ``length`` and ``differential`` entries are keyed
        ``<query_id>#<repeat>``.
        """
        repeats = min(len(baselines), len(candidates))
        keyed = repeats > 1
        # (pair key, query id, baseline result, candidate result)
        pairs: list[tuple[str, str, EvaluationResult, EvaluationResult]] = []
        for index in range(repeats):
            by_id = {r.query_id: r for r in baselines[index]}
            pairs.extend(
                (
                    f"{r.query_id}#{index + 1}" if keyed else r.query_id,
                    r.query_id,
                    by_id[r.query_id],
                    r,
                )
                for r in candidates[index]
                if r.query_id in by_id
            )
        query_of = {key: qid for key, qid, _, _ in pairs}
        # Per pair and criterion, the candidate's score: 1, 0.5 or 0, and in
        # the differential mode the mean over its two split slots.
        scores: dict[str, dict[str, float]] = {}
        errors: Counter[str] = Counter()
        to_judge: list[tuple[str, EvaluationResult, EvaluationResult]] = []
        for key, _, base, cand in pairs:
            if _unanswerable(base) or _unanswerable(cand):
                errors["unanswerable_skipped"] += 1
                continue
            base_ok, cand_ok = _answered(base), _answered(cand)
            if base_ok and cand_ok:
                to_judge.append((key, base, cand))
            elif base_ok:
                errors["candidate_errors"] += 1
                scores[key] = dict.fromkeys(self.criteria, _SCORES["baseline"])
            elif cand_ok:
                errors["baseline_errors"] += 1
                scores[key] = dict.fromkeys(self.criteria, _SCORES["candidate"])
            else:
                errors["both_errors"] += 1
        # The judge calls nest under one span instead of each starting a root
        # trace, as the other evaluators' do.
        with observability.span(
            "kg_rag.evaluate", evaluator="pairwise_judge", items=len(to_judge)
        ):
            judged = await asyncio.gather(
                *(self._safe_judge(b, c) for _, b, c in to_judge)
            )
        agreements: list[dict[str, bool]] = []
        judged_keys: list[str] = []
        identical = 0
        lengths: dict[str, dict[str, Any]] = {}
        splits: dict[str, dict[str, Any]] = {}
        # Differential mode: per split slot, each judged pair's scores.
        slot_scores: dict[str, dict[str, dict[str, float]]] = {
            slot: {} for slot in SPLIT_SLOTS
        }
        for (key, base, cand), verdict in zip(to_judge, judged, strict=True):
            if verdict is None:
                continue
            slots, pair_splits = verdict
            judged_keys.append(key)
            agreements.extend(agreed for _, agreed in slots if agreed is not None)
            identical += all(agreed is None for _, agreed in slots)
            scores[key] = {
                c: float(np.mean([_SCORES[outcome[c]] for outcome, _ in slots]))
                for c in self.criteria
            }
            lengths[key] = answer_lengths(_words(base), _words(cand))
            if pair_splits is not None:
                for slot, (outcome, _) in zip(SPLIT_SLOTS, slots, strict=True):
                    slot_scores[slot][key] = {
                        c: _SCORES[outcome[c]] for c in self.criteria
                    }
                for side in ("baseline", "candidate"):
                    lengths[key][f"{side}_unique_tokens"] = _mean_tokens(
                        split[side] for split in pair_splits.values()
                    )
                lengths[key]["shared_tokens"] = _mean_tokens(
                    split["shared"] for split in pair_splits.values()
                )
                splits[key] = {
                    slot: {**pair_splits[slot], "outcome": outcome}
                    for slot, (outcome, _) in zip(SPLIT_SLOTS, slots, strict=True)
                }
        # Each pair's outcome: the side its score favours.
        outcomes = {
            key: {c: _outcome(score) for c, score in pair.items()}
            for key, pair in scores.items()
        }
        criteria: dict[str, Any] = {}
        for criterion in self.criteria:
            counts = Counter(o[criterion] for o in outcomes.values())
            # Ties split evenly, as the papers report win rates; each
            # query's repeats averaged first (one score per query).
            by_query = _query_scores(scores, query_of, criterion)
            values = list(by_query.values())
            rate, low, high, p_value = None, None, None, None
            if judged_keys:
                rate = float(np.mean(values))
                low, high = bootstrap_ci(values, **_BOOTSTRAP)
                # Unadjusted; adjust_for_multiplicity adds the Holm value.
                p_value = sign_flip_p_value(
                    [v - _SCORES["tie"] for v in values],
                    resamples=_BOOTSTRAP["resamples"],
                    seed=_BOOTSTRAP["seed"],
                )
            criteria[criterion] = {
                "candidate_wins": counts["candidate"],
                "baseline_wins": counts["baseline"],
                "ties": counts["tie"],
                "candidate_win_rate": rate,
                "ci_low": low,
                "ci_high": high,
                "p_value": p_value,
                "decided": counts["candidate"] + counts["baseline"],
                "position_consistency": (
                    sum(a[criterion] for a in agreements) / len(agreements)
                    if agreements
                    else None
                ),
                # Per query, the candidate's score (repeats and slots
                # averaged): what the win rate, interval and p-value read.
                "per_query": dict(sorted(by_query.items())) if judged_keys else {},
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
        per_query: dict[str, Any]
        if keyed:
            per_query = {
                qid: [outcomes.get(f"{qid}#{index + 1}") for index in range(repeats)]
                for qid in sorted(set(query_of.values()))
                if any(f"{qid}#{index + 1}" in outcomes for index in range(repeats))
            }
        else:
            per_query = dict(sorted(outcomes.items()))
        result: dict[str, Any] = {
            "mode": self.mode,
            "repeats": repeats,
            "pairs": len(pairs),
            "judged": len(judged_keys),
            "failed": sum(v is None for v in judged),
            "candidate_errors": errors["candidate_errors"],
            "baseline_errors": errors["baseline_errors"],
            "both_errors": errors["both_errors"],
            "unanswerable_skipped": errors["unanswerable_skipped"],
            "position_consistency": consistency,
            "bootstrap": dict(_BOOTSTRAP),
            "criteria": criteria,
            "per_query": per_query,
            # Judged pairs only: a pair decided by a failed answer has no
            # length to compare.
            "length": length_summary(
                lengths, {k: outcomes[k] for k in judged_keys}, self.criteria
            ),
        }
        if self.mode == "differential":
            result["identical_pairs"] = identical
            result["differential"] = {"per_query": dict(sorted(splits.items()))}
            result["slots"] = _slot_summary(slot_scores, query_of, self.criteria)
        return result

    async def _safe_judge(
        self, baseline: EvaluationResult, candidate: EvaluationResult
    ) -> _Judged | None:
        try:
            if self.mode == "differential":
                return await self._ajudge_differential(
                    candidate.question, _words(baseline), _words(candidate)
                )
            outcome, consistent = await self._ajudge_both_orders(
                candidate.question, _words(baseline), _words(candidate)
            )
            return [(outcome, consistent)], None
        except Exception as e:  # noqa: BLE001 - a failed pair is counted, not fatal
            # The error may quote the judge's reply: a digest at WARNING.
            logger.warning(
                "Pairwise judge failed for '%s': %s",
                candidate.query_id,
                error_digest(e),
            )
            logger.debug("Pairwise judge error for '%s': %s", candidate.query_id, e)
            return None


def _outcome(score: float) -> str:
    """The side a pair's score favours (``"tie"`` at 0.5)."""
    if score > _SCORES["tie"]:
        return "candidate"
    return "baseline" if score < _SCORES["tie"] else "tie"


def _query_scores(
    scores: Mapping[str, Mapping[str, float]],
    query_of: Mapping[str, str],
    criterion: str,
) -> dict[str, float]:
    """Per query, the mean of its pairs' (repeats') ``criterion`` scores."""
    by_query: dict[str, list[float]] = {}
    for key, pair in scores.items():
        by_query.setdefault(query_of[key], []).append(pair[criterion])
    return {q: float(np.mean(v)) for q, v in by_query.items()}


def _slot_summary(
    slot_scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    query_of: Mapping[str, str],
    criteria: Sequence[str],
) -> dict[str, Any]:
    """The differential mode's slot effect per criterion: the candidate's
    win rate over the judged pairs when the split read it as Answer 2
    (``baseline_first``) and as Answer 1 (``candidate_first``), and the
    first minus the second (``slot_effect``: 0 for a split model without a
    slot preference; the reported win rate averages both)."""
    summary: dict[str, Any] = {}
    for criterion in criteria:
        rates = {
            slot: (
                float(
                    np.mean(list(_query_scores(by_pair, query_of, criterion).values()))
                )
                if by_pair
                else None
            )
            for slot, by_pair in slot_scores.items()
        }
        first, second = (rates[slot] for slot in SPLIT_SLOTS)
        summary[criterion] = {
            **{f"{slot}_win_rate": rate for slot, rate in rates.items()},
            "slot_effect": (
                first - second if first is not None and second is not None else None
            ),
        }
    return {"order": list(SPLIT_SLOTS), "criteria": summary}


def _mean_tokens(statement_lists: Iterable[Sequence[str]]) -> float:
    """The mean estimated tokens of the statement lists (one per split)."""
    counts = [estimate_token_count("\n".join(s)) for s in statement_lists]
    return float(np.mean(counts)) if counts else 0.0


def _words(result: EvaluationResult) -> str:
    """The answer the judge reads: its words, without citation markers
    ([1]), so a strategy that cites (search.citations) is judged, and
    measured for length, on what it says like one that does not."""
    return strip_markers(result.generated_answer or "")


def _answered(result: EvaluationResult) -> bool:
    return bool(result.generated_answer) and not result.error


def _unanswerable(result: EvaluationResult) -> bool:
    return result.metadata.get("answerable") is False
