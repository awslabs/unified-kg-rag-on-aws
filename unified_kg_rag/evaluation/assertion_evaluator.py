# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assertion evaluator: does the answer support the item's assertions?

A dataset item may carry ``assertions`` (``EvaluationAssertion``):
statements a good answer supports (``kind: positive``) or must not state
(``kind: negative``, e.g. a known hallucination). ``prepare-benchmark
assertions`` generates positive ones from a corpus, in the style of
BenchmarkQED's AutoE assertion scoring (https://github.com/microsoft/benchmark-qed,
MIT); negative ones are written by hand.

Per answer and trial, one judge call (``models.roles.evaluation``,
``AssertionJudgementPrompt``) labels every assertion of the item
``supported``, ``contradicted`` or ``absent``; the assertions are shuffled
per trial with a seed of the query id, so their order is no position bias,
and the labels are mapped back. ``evaluation.assertion_trials`` trials run
per answer, each in an order no earlier trial of the query used while one is
left (``n`` assertions have ``n!`` orders). A deterministic judge (temperature
0) gives the same labels for the same order, so an item with fewer orders
than trials (one assertion, or two with three trials) repeats a call that
adds no information; the report records ``distinct_orders``. The judge calls
of all queries and trials share ``processing.max_concurrency``. An assertion
takes a label when more than half of the successful trials gave it (a
majority vote: BenchmarkQED's pass rule, a trial mean above 0.5). Per query:

- ``assertion_recall``: positive assertions supported / positive assertions.
- ``assertion_contradiction_rate``: positive assertions contradicted /
  positive assertions (lower is better).
- ``negative_assertion_violation_rate``: negative assertions supported /
  negative assertions (lower is better).

A trial whose call fails, or whose reply misses an assertion, repeats one or
uses another label, is a failed trial; the metrics read the trials that
succeeded (``failed_trials`` in the report). A query whose trials all failed
records its metrics under ``failed_metrics``, never as 0. A query without
assertions skips every metric, one without negative (or positive) ones the
metrics of that kind. The labels are fixed tokens in a JSON reply, so the
parser reads the reply the same way whatever the language of the question,
answer and assertions.
"""

from __future__ import annotations

import asyncio
import math
import random
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from unified_kg_rag.adapters.evaluators.base import (
    FAILED_METRICS_KEY,
    SKIPPED_METRICS_KEY,
    BaseGraphRAGEvaluator,
    error_digest,
    judge_answers,
    judge_model_kwargs,
)
from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import (
    Config,
    EvaluationAssertion,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
    ModelPurpose,
    ModelRole,
)
from unified_kg_rag.domain.prompts import ASSERTION_VERDICTS, AssertionJudgementPrompt
from unified_kg_rag.shared import EvaluationException, get_logger

logger = get_logger(__name__)

# Where EvaluationManager threads an item's assertions onto its result.
ASSERTIONS_KEY = "assertions"
SKIP_REASON_NO_ASSERTIONS = "no_assertions"
SKIP_REASON_NO_POSITIVE = "no_positive_assertions"
SKIP_REASON_NO_NEGATIVE = "no_negative_assertions"
# An assertion takes a label when more than this share of the trials gave it.
PASS_THRESHOLD = 0.5

_POSITIVE_METRICS = (
    EvaluationMetricType.ASSERTION_RECALL,
    EvaluationMetricType.ASSERTION_CONTRADICTION_RATE,
)
_NEGATIVE_METRICS = (EvaluationMetricType.NEGATIVE_ASSERTION_VIOLATION_RATE,)


def assertions_of(metadata: dict[str, Any]) -> list[EvaluationAssertion]:
    """The item's assertions as ``EvaluationAssertion`` (a bare string is a
    positive one).

    Raises:
        EvaluationException: An entry is neither a string nor an assertion
            object.
    """
    raw = metadata.get(ASSERTIONS_KEY) or []
    if not isinstance(raw, list):
        raise EvaluationException("'assertions' must be a list")
    parsed: list[EvaluationAssertion] = []
    for index, entry in enumerate(raw):
        try:
            parsed.append(
                EvaluationAssertion(text=entry)
                if isinstance(entry, str)
                else EvaluationAssertion.model_validate(entry)
            )
        except ValidationError as e:
            raise EvaluationException(f"assertion {index}: {e}") from e
    return parsed


def trial_order(query_id: str, trial: int, count: int) -> list[int]:
    """The order the assertions are shown in for one trial: a permutation of
    ``range(count)`` seeded by the query id (the same in every run and
    process). The trials of a query take distinct orders until all ``count!``
    are used, then repeat them in the same sequence."""
    distinct = math.factorial(count) if count < 13 else None  # 13! > 6e9
    cycle = trial if distinct is None else trial % distinct
    rng = random.Random(f"{query_id}\0order")  # nosec B311 - not crypto
    seen: list[list[int]] = []
    while len(seen) <= cycle:
        order = list(range(count))
        rng.shuffle(order)
        if order not in seen:
            seen.append(order)
    return seen[cycle]


def distinct_orders(count: int, trials: int) -> int:
    """How many of ``trials`` trials show ``count`` assertions in an order no
    other trial used."""
    return trials if count >= 13 else min(trials, math.factorial(count))


def parse_verdicts(output: Any, count: int) -> list[str]:
    """The labels of a judge reply for ``count`` assertions, by shown number.

    Raises:
        EvaluationException: The reply has no ``verdicts`` list, an id is not
            one of ``1..count`` or repeats, an assertion has no label, or a
            label is not one of ``ASSERTION_VERDICTS``.
    """
    entries = output.get("verdicts") if isinstance(output, dict) else None
    if not isinstance(entries, list):
        raise EvaluationException(f"no 'verdicts' list in the judge reply: {output!r}")
    labels: dict[int, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise EvaluationException(f"a verdict is not an object: {entry!r}")
        try:
            number = int(str(entry.get("id")).strip())
        except ValueError as e:
            raise EvaluationException(f"verdict id {entry.get('id')!r}") from e
        label = str(entry.get("verdict", "")).strip().strip('"').lower()
        if not 1 <= number <= count:
            raise EvaluationException(f"verdict id {number} is not 1..{count}")
        if label not in ASSERTION_VERDICTS:
            raise EvaluationException(f"assertion {number}: label {label!r}")
        if number in labels:
            raise EvaluationException(f"assertion {number} is labelled twice")
        labels[number] = label
    missing = [n for n in range(1, count + 1) if n not in labels]
    if missing:
        raise EvaluationException(f"no label for assertion(s) {missing}")
    return [labels[n] for n in range(1, count + 1)]


def aggregate_trials(
    kinds: Sequence[str], trials: Sequence[Sequence[str]]
) -> dict[str, Any]:
    """Per-assertion label shares over ``trials`` (each a label per assertion,
    in the assertions' order) and the query's metrics: an assertion takes a
    label when more than ``PASS_THRESHOLD`` of the trials gave it. A metric
    whose kind has no assertion is ``None``."""
    shares = [
        {
            label: sum(trial[i] == label for trial in trials) / len(trials)
            for label in ASSERTION_VERDICTS
        }
        for i in range(len(kinds))
    ]
    positive = [s for s, kind in zip(shares, kinds, strict=True) if kind == "positive"]
    negative = [s for s, kind in zip(shares, kinds, strict=True) if kind == "negative"]

    def rate(group: list[dict[str, float]], label: str) -> float | None:
        if not group:
            return None
        return sum(s[label] > PASS_THRESHOLD for s in group) / len(group)

    return {
        "shares": shares,
        EvaluationMetricType.ASSERTION_RECALL.value: rate(positive, "supported"),
        EvaluationMetricType.ASSERTION_CONTRADICTION_RATE.value: rate(
            positive, "contradicted"
        ),
        EvaluationMetricType.NEGATIVE_ASSERTION_VIOLATION_RATE.value: rate(
            negative, "supported"
        ),
    }


def _numbered(assertions: Sequence[EvaluationAssertion], order: Sequence[int]) -> str:
    return "\n".join(
        f"{number}. {assertions[index].text}"
        for number, index in enumerate(order, start=1)
    )


class AssertionEvaluator(BaseGraphRAGEvaluator):
    """Scores an answer by the item's assertions it supports or states."""

    def __init__(
        self,
        config: Config,
        *,
        providers: Providers | None = None,
        show_progress: bool = True,
    ) -> None:
        self.providers = Providers.resolve(config, providers)
        super().__init__(
            config=config,
            evaluator_type=EvaluatorType.ASSERTION,
            show_progress=show_progress,
        )

    def _initialize_evaluator(self, **kwargs: Any) -> None:
        self.trials = self.config.evaluation.assertion_trials
        # One limit for every judge call: the batch runs max_concurrency
        # queries at once and each its trials at once, so without it the
        # calls in flight were max_concurrency x assertion_trials.
        self._slots: tuple[asyncio.AbstractEventLoop, asyncio.Semaphore] | None = None
        self.chain = setup_chain(
            self.providers.llm_factory,
            self.config.model_id(ModelRole.EVALUATION),
            AssertionJudgementPrompt,
            judge_answers(self.providers.llm_factory, self.config, ("verdicts",)),
            custom_prompts=self.config.custom_prompts,
            model_purpose=ModelPurpose.EVALUATION,
            **judge_model_kwargs(self.config),
        )

    def metric_types(self) -> list[EvaluationMetricType]:
        return [*_POSITIVE_METRICS, *_NEGATIVE_METRICS]

    def _judge_slot(self) -> asyncio.Semaphore:
        """The semaphore bounding judge calls, one per event loop (a
        semaphore is bound to the loop it first waits in)."""
        loop = asyncio.get_running_loop()
        if self._slots is None or self._slots[0] is not loop:
            self._slots = (
                loop,
                asyncio.Semaphore(self.config.processing.max_concurrency),
            )
        return self._slots[1]

    async def _trial(
        self,
        result: EvaluationResult,
        assertions: Sequence[EvaluationAssertion],
        trial: int,
    ) -> list[str]:
        """One trial's label per assertion, in the assertions' own order."""
        order = trial_order(result.query_id, trial, len(assertions))
        async with self._judge_slot():
            output = await self.chain.ainvoke(
                {
                    "question": result.question,
                    "answer": self.answer_words(result),
                    "assertions": _numbered(assertions, order),
                }
            )
        shown = parse_verdicts(output, len(assertions))
        labels = [""] * len(assertions)
        for position, index in enumerate(order):
            labels[index] = shown[position]
        return labels

    async def aevaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        assertions = assertions_of(result.metadata)
        kinds = [a.kind for a in assertions]
        skipped: dict[str, str] = {}
        if "positive" not in kinds:
            reason = SKIP_REASON_NO_ASSERTIONS if not kinds else SKIP_REASON_NO_POSITIVE
            skipped.update({m.value: reason for m in _POSITIVE_METRICS})
        if "negative" not in kinds:
            reason = SKIP_REASON_NO_ASSERTIONS if not kinds else SKIP_REASON_NO_NEGATIVE
            skipped.update({m.value: reason for m in _NEGATIVE_METRICS})
        if not assertions:
            return self._report(query, [], {SKIPPED_METRICS_KEY: skipped})
        outcomes = await asyncio.gather(
            *(self._trial(result, assertions, t) for t in range(self.trials)),
            return_exceptions=True,
        )
        trials: list[list[str]] = []
        errors: list[Exception] = []
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                if not isinstance(outcome, Exception):
                    raise outcome  # a cancellation is not a failed trial
                errors.append(outcome)
            else:
                trials.append(outcome)
        attempted = [m for m in self.metric_types() if m.value not in skipped]
        metadata: dict[str, Any] = {
            **self._extract_search_metadata(result),
            "assertions": len(assertions),
            "positive_assertions": kinds.count("positive"),
            "negative_assertions": kinds.count("negative"),
            "trials": self.trials,
            "distinct_orders": distinct_orders(len(assertions), self.trials),
            "failed_trials": len(errors),
        }
        if skipped:
            metadata[SKIPPED_METRICS_KEY] = skipped
        # The error may quote the judge's reply: a digest at WARNING, the
        # text at DEBUG.
        if not trials:
            logger.warning(
                "Assertion judge failed every trial for '%s': %s",
                query.query_id,
                error_digest(errors[-1]) if errors else "no trial ran",
            )
            if errors:
                logger.debug(
                    "Assertion judge error for '%s': %s", query.query_id, errors[-1]
                )
            last = (str(errors[-1]) or type(errors[-1]).__name__) if errors else ""
            reason = f"Assertion judge failed: {last or 'no trial'}"
            metadata[FAILED_METRICS_KEY] = {m.value: reason for m in attempted}
            return self._report(query, [], metadata)
        if errors:
            logger.warning(
                "Assertion judge failed %s of %s trials for '%s' (scored on the "
                "rest): %s",
                len(errors),
                self.trials,
                query.query_id,
                error_digest(errors[-1]),
            )
            logger.debug(
                "Assertion judge error for '%s': %s", query.query_id, errors[-1]
            )
        aggregate = aggregate_trials(kinds, trials)
        metadata["assertion_labels"] = [
            {
                "text": a.text,
                "kind": a.kind,
                "shares": shares,
            }
            for a, shares in zip(assertions, aggregate["shares"], strict=True)
        ]
        metrics = [
            EvaluationMetric(metric_type=m, value=float(aggregate[m.value]))
            for m in attempted
            if aggregate[m.value] is not None
        ]
        return self._report(query, metrics, metadata)

    def _report(
        self,
        query: EvaluationQuery,
        metrics: list[EvaluationMetric],
        metadata: dict[str, Any],
    ) -> EvaluationReport:
        return EvaluationReport(
            query_id=query.query_id,
            evaluator_type=self.evaluator_type,
            metrics=metrics,
            evaluation_time=datetime.now(),
            metadata=metadata,
        )
