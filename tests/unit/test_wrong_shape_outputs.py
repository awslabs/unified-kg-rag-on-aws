# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output that parses but has the wrong shape goes to the output fixer.

Graph extraction, claims and gleaning checked the shape of an answer after
the chain, outside ``XMLOutputFixingParser``: an extraction whose sections
said "None" or that one ``<output>`` element wrapped failed the unit after
every retry without a repair, a gleaning answer without its
``<refinement_plan>`` root was read as "nothing to add" and the unit marked
gleaned for good, and a claims answer in prose was counted as cut off (never
retried). The parsers now unwrap one wrapper, read a section that should
list elements but holds text only as empty (in any language: "None",
"없음", "Keine"), and raise ``OutputParserException`` for any other wrong shape, which
the fixer repairs. The chains run for real on a scripted model.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from unified_kg_rag.adapters.ingestion.claim_extractor import ClaimExtractor
from unified_kg_rag.adapters.ingestion.gleaner import GraphGleaner
from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import Config, Entity, TextUnit, TextUnitExtraction

pytestmark = pytest.mark.unit

_ENTITY = (
    "<entity><name>Acme</name><type>ORGANIZATION</type><description>d"
    "</description><confidence>8</confidence><source_text>Acme</source_text>"
    "</entity>"
)
_FIXED_EXTRACTION = f"<entities>{_ENTITY}</entities>\n<relationships></relationships>"
_CLAIM = (
    "<claim><subject>Vendor</subject><object>Buyer</object><claim_type>"
    "FINANCIAL</claim_type><claim_status>TRUE</claim_status><description>d"
    "</description><source_text>Vendor pays Buyer</source_text></claim>"
)
_ISSUE = (
    "<issue><issue_type>MISSING_ENTITY</issue_type><details><name>Depot</name>"
    "<type>LOCATION</type><description>a depot</description></details>"
    "<text_evidence>Depot stores widgets</text_evidence></issue>"
)
_PLAN = f"<identified_issues>{_ISSUE}</identified_issues>"


def _providers(config: Config, answer: str, fixed: str, calls: list[str]) -> Providers:
    """Providers whose model gives ``answer``, and ``fixed`` when asked to
    repair one (the output fixer's prompt), recording which was asked."""

    def respond(system: str, human: str) -> str:
        repair = "did not satisfy the constraints" in system + human
        calls.append("fixer" if repair else "model")
        return fixed if repair else answer

    return Providers(
        config, boto_session=MagicMock(), llm_factory=ScriptedLLMFactory(respond)
    )


@pytest.fixture
def config() -> Config:
    config = Config()
    config.retry.max_attempts = 1
    config.processing.graph_extraction.entity_grounding.enabled = False
    return config


def _extract(config: Config, answer: str) -> tuple[list[str], list[str], list[str]]:
    """(calls, entity names, failed units) of one unit's graph extraction."""
    calls: list[str] = []
    extractor = GraphExtractor(
        config, providers=_providers(config, answer, _FIXED_EXTRACTION, calls)
    )
    records, stats = extractor.extract_units([TextUnit(id="t1", text="Acme")])
    names = [e.name for record in records for e in record.entities]
    return calls, names, stats.failed_text_unit_ids


@pytest.mark.parametrize(
    "answer",
    [
        "<entities>None</entities>\n<relationships>None</relationships>",
        "<entities>N/A</entities>",
        # Text in the model's language: no word list recognised these.
        "<entities>없음</entities>\n<relationships>없음</relationships>",
        "<entities>なし</entities>\n<relationships>无</relationships>",
        "<entities>Keine</entities>\n<relationships>Нет</relationships>",
        "<entities>No entities in this text.</entities>\n<relationships/>",
    ],
)
def test_an_extraction_saying_none_is_empty(config: Config, answer: str) -> None:
    assert _extract(config, answer) == (["model"], [], [])


@pytest.mark.parametrize("none", ["None", "없음", "Aucune", "No relationships found."])
def test_a_none_relationships_section_keeps_the_entities(
    config: Config, none: str
) -> None:
    answer = f"<entities>{_ENTITY}</entities>\n<relationships>{none}</relationships>"
    assert _extract(config, answer) == (["model"], ["Acme"], [])


def test_a_wrapped_extraction_is_unwrapped(config: Config) -> None:
    answer = f"<output>\n{_FIXED_EXTRACTION}\n</output>"
    assert _extract(config, answer) == (["model"], ["Acme"], [])


@pytest.mark.parametrize(
    "answer",
    [
        # Bullets, entities named by an attribute.
        "Nothing to extract:\n- a list of numbers\n- no entities",
        '<entities><entity name="Acme" type="ORG"/></entities>\n<relationships/>',
    ],
)
def test_a_wrong_extraction_shape_is_repaired(config: Config, answer: str) -> None:
    assert _extract(config, answer) == (["model", "fixer"], ["Acme"], [])


def _claims(config: Config, answer: str) -> tuple[list[str], list[str], list[str]]:
    config.processing.claim_extraction.enabled = True
    calls: list[str] = []
    extractor = ClaimExtractor(
        config,
        providers=_providers(config, answer, f"<claims>{_CLAIM}</claims>", calls),
    )
    claims, stats = extractor.extract_units([TextUnit(id="t1", text="Vendor")])
    subjects = [c.subject_name for unit in claims.values() for c in unit]
    return calls, subjects, stats.failed_text_unit_ids


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("<claims>None</claims>", (["model"], [], [])),
        (f"<output><claims>{_CLAIM}</claims></output>", (["model"], ["Vendor"], [])),
        # It raised LLMOutputTruncatedError: counted as cut off, never repaired.
        ("No claims:\n- nothing factual", (["model", "fixer"], ["Vendor"], [])),
        ("<claims>There are none here.</claims>", (["model"], [], [])),
        ("<claims>없음</claims>", (["model"], [], [])),
    ],
)
def test_a_wrong_claims_shape_is_unwrapped_or_repaired(
    config: Config, answer: str, expected: tuple
) -> None:
    assert _claims(config, answer) == expected


def _glean(config: Config, answer: str) -> tuple[list[str], list[str], bool]:
    """(calls, entity names, gleaned) of one unit's first gleaning round."""
    config.processing.gleaning.max_rounds = 1
    calls: list[str] = []
    gleaner = GraphGleaner(
        config,
        show_progress=False,
        providers=_providers(
            config, answer, f"<refinement_plan>{_PLAN}</refinement_plan>", calls
        ),
    )
    unit = TextUnit(id="t1", text="Vendor supplies widgets. Depot stores widgets.")
    vendor = Entity(id="v", name="Vendor", type="ORGANIZATION", text_unit_ids=["t1"])
    records, _ = gleaner.glean(
        [unit], [TextUnitExtraction(text_unit_id="t1", entities=[vendor])]
    )
    return calls, [e.name for e in records[0].entities], records[0].gleaned


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        # Root-less: its issues were ignored and the unit marked gleaned.
        (_PLAN, (["model"], ["Vendor", "Depot"], True)),
        (
            f"<response><refinement_plan>{_PLAN}</refinement_plan></response>",
            (["model"], ["Vendor", "Depot"], True),
        ),
        ("<refinement_plan>None</refinement_plan>", (["model"], ["Vendor"], True)),
        ("<refinement_plan>なし</refinement_plan>", (["model"], ["Vendor"], True)),
        (
            "The extraction looks complete to me.",
            (["model", "fixer"], ["Vendor", "Depot"], True),
        ),
        (
            "<quality_scores><completeness_score>1</completeness_score>"
            "</quality_scores>",
            (["model", "fixer"], ["Vendor", "Depot"], True),
        ),
    ],
)
def test_a_wrong_gleaning_shape_is_read_or_repaired(
    config: Config, answer: str, expected: tuple
) -> None:
    assert _glean(config, answer) == expected
