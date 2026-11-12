# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The regression corpus (``tests/fixtures/language_cases.py``) through the
language-aware readers: a query's language, the gold number an answer
matches, an answer's containment of its gold, the router's strategy name
and a stored name's cover of a mention.

Each case is pinned to its stated answer, or to a documented trade-off's;
a rule added for one case that moves another fails here.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import BaseModel

from tests.fixtures.language_cases import (
    CONTAINS,
    COVERS,
    DETECTION,
    NUMBERS,
    ROUTER,
    Case,
    corpus_score,
)
from unified_kg_rag.domain.languages import text_language
from unified_kg_rag.domain.models import SearchStrategy
from unified_kg_rag.domain.retrieval.linking import covers
from unified_kg_rag.evaluation.answer_match_evaluator import (
    answer_contains,
    number_matches,
)
from unified_kg_rag.evaluation.text_matching import stated_numbers
from unified_kg_rag.shared import ModelOutputError
from unified_kg_rag.shared.utils import named_choice, parse_model_object

pytestmark = pytest.mark.unit

_STRATEGIES = frozenset(strategy.value for strategy in SearchStrategy)


class _Route(BaseModel):
    strategy: str


def detected(case: Case) -> object:
    return text_language(case.text, case.context or "en").value


def counted(case: Case) -> object:
    """The gold the answer matches as ``answer_match`` scores a number item
    (``number_matches``): the stated one when it does, else the numbers the
    answer states, sorted (``None``: none)."""
    gold = case.expected
    if gold is not None and number_matches(
        case.text, str(gold), case.context, case.question
    ):
        return gold
    return tuple(sorted(stated_numbers(case.text, case.context))) or None


def contained(case: Case) -> object:
    """The score ``answer_match`` gives the answer against its gold
    (``answer_contains``)."""
    assert case.gold is not None
    return answer_contains(case.text, case.gold, case.context)


def routed(case: Case) -> object:
    """The strategy the router reads without a repair call: a name line
    (``named_choice``), else the JSON reader's; ``None`` needs the
    repair."""
    if (name := named_choice(case.text, _STRATEGIES)) is not None:
        return name
    try:
        return parse_model_object(case.text, _Route).strategy.strip().casefold()
    except ModelOutputError:
        return None


def covered(case: Case) -> object:
    return covers(case.text, case.context or "")


def _ids(cases: tuple[Case, ...]) -> list[str]:
    return [f"{i}-{case.context}" for i, case in enumerate(cases)]


@pytest.mark.parametrize("case", DETECTION, ids=_ids(DETECTION))
def test_detection(case: Case) -> None:
    assert detected(case) == case.answer, case.text


@pytest.mark.parametrize("case", NUMBERS, ids=_ids(NUMBERS))
def test_numbers(case: Case) -> None:
    assert counted(case) == case.answer, case.text


@pytest.mark.parametrize("case", CONTAINS, ids=_ids(CONTAINS))
def test_contains(case: Case) -> None:
    assert contained(case) == case.answer, (case.text, case.gold)


@pytest.mark.parametrize("case", ROUTER, ids=_ids(ROUTER))
def test_router(case: Case) -> None:
    assert routed(case) == case.answer, case.text


@pytest.mark.parametrize("case", COVERS, ids=_ids(COVERS))
def test_covers(case: Case) -> None:
    assert covered(case) == case.answer, (case.text, case.context)


@pytest.mark.parametrize(
    ("cases", "read"),
    [
        (DETECTION, detected),
        (NUMBERS, counted),
        (CONTAINS, contained),
        (ROUTER, routed),
        (COVERS, covered),
    ],
    ids=["detection", "numbers", "contains", "router", "covers"],
)
def test_every_miss_is_a_documented_tradeoff(
    cases: tuple[Case, ...], read: Callable[[Case], object]
) -> None:
    # A trade-off answers otherwise than stated, and every other case as
    # stated: a fixed trade-off is dropped from the corpus, not kept.
    for case in cases:
        if case.tradeoff is not None:
            assert case.tradeoff.answer != case.expected, case.text
    passed, total = corpus_score(cases, read)
    assert total - passed == sum(case.tradeoff is not None for case in cases)
