# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each LLM chain runs on the model of its own ``models.roles`` entry.

Every role is pinned to its own probe model id, and the chat model inside each
chain is read back: a component that built a chain from another role's entry
(two roles swapped in one constructor) would still pass any test that only
checks that both roles were requested.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable

from tests.fixtures.fakes.chat_models import fake_model_info
from unified_kg_rag.adapters.ingestion.claim_extractor import ClaimExtractor
from unified_kg_rag.adapters.ingestion.community_detector import CommunityDetector
from unified_kg_rag.adapters.ingestion.description_summarizer import (
    DescriptionSummarizer,
)
from unified_kg_rag.adapters.ingestion.gleaner import GraphGleaner
from unified_kg_rag.adapters.ingestion.graph_extractor import GraphExtractor
from unified_kg_rag.adapters.ingestion.translator import TextUnitTranslator
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.search_strategies.drift_search import (
    DriftSearchStrategy,
)
from unified_kg_rag.adapters.search_strategies.global_search import (
    GlobalSearchStrategy,
)
from unified_kg_rag.domain.models import Config, ModelRole

pytestmark = pytest.mark.unit


class _Probe(BaseChatModel):
    """A chat model that only remembers which model id it was built for."""

    model_name: str

    @property
    def _llm_type(self) -> str:
        return "probe"

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise AssertionError("a probe model is never called")


class _ProbeFactory:
    def get_model(self, model_id: Any, **kwargs: Any) -> _Probe:
        return _Probe(model_name=str(model_id))

    def get_model_info(self, model_id: Any) -> Any:
        return fake_model_info(model_id)


def _model_of(chain: Runnable) -> str:
    """The model id of the chat model inside ``prompt | llm | ... | parser``."""
    pending: list[Any] = [chain]
    while pending:
        node = pending.pop()
        if isinstance(node, _Probe):
            return node.model_name
        pending.extend(getattr(node, "steps", None) or [])
        for attribute in ("bound", "runnable", "first", "last"):
            if (child := getattr(node, attribute, None)) is not None:
                pending.append(child)
    raise AssertionError(f"no chat model in {chain!r}")


def _probe_config() -> Config:
    config = Config()
    for role in ModelRole:
        setattr(config.models.roles, role.value, f"probe-{role.value}")
    config.processing.translation.enabled = True
    config.processing.gleaning.enabled = True
    config.processing.claim_extraction.enabled = True
    config.graph.community_detection.report_generation.enabled = True
    config.search.global_search.use_dynamic_selection = True
    config.search.global_search.reduce_with_llm = True
    return config


Builder = Callable[[Config, Providers], Any]

CASES: list[tuple[str, Builder, str, ModelRole]] = [
    (
        "extraction",
        lambda c, p: GraphExtractor(c, providers=p),
        "graph_extractor",
        ModelRole.GRAPH_EXTRACTION,
    ),
    (
        "gleaning",
        lambda c, p: GraphGleaner(c, show_progress=False, providers=p),
        "graph_refiner",
        ModelRole.GLEANING,
    ),
    (
        "claims",
        lambda c, p: ClaimExtractor(c, providers=p),
        "claim_extractor",
        ModelRole.CLAIM_EXTRACTION,
    ),
    (
        "summary",
        lambda c, p: DescriptionSummarizer(c, providers=p),
        "summarizer",
        ModelRole.DESCRIPTION_SUMMARY,
    ),
    (
        "translation",
        lambda c, p: TextUnitTranslator(c, show_progress=False, providers=p),
        "translator",
        ModelRole.TRANSLATION,
    ),
    (
        "report",
        lambda c, p: CommunityDetector(c, show_progress=False, providers=p),
        "report_generator",
        ModelRole.COMMUNITY_REPORT,
    ),
    (
        "relevance",
        lambda c, p: GlobalSearchStrategy(c, {}, providers=p),
        "community_relevance_scorer",
        ModelRole.COMMUNITY_RELEVANCE,
    ),
    (
        "map",
        lambda c, p: GlobalSearchStrategy(c, {}, providers=p),
        "map_rater",
        ModelRole.GLOBAL_MAP,
    ),
    (
        "reduce",
        lambda c, p: GlobalSearchStrategy(c, {}, providers=p),
        "map_reducer",
        ModelRole.GLOBAL_REDUCE,
    ),
    (
        "hyde",
        lambda c, p: DriftSearchStrategy(c, {}, providers=p),
        "hyde",
        ModelRole.DRIFT_PRIMER,
    ),
    (
        "primer",
        lambda c, p: DriftSearchStrategy(c, {}, providers=p),
        "primer",
        ModelRole.DRIFT_PRIMER,
    ),
    (
        "action",
        lambda c, p: DriftSearchStrategy(c, {}, providers=p),
        "action",
        ModelRole.DRIFT_ACTION,
    ),
]


@pytest.mark.parametrize(
    ("builder", "attribute", "role"),
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_each_chain_runs_on_its_roles_model(
    builder: Builder, attribute: str, role: ModelRole
) -> None:
    config = _probe_config()
    component = builder(config, Providers(config, llm_factory=_ProbeFactory()))
    assert _model_of(getattr(component, attribute)) == f"probe-{role.value}"
