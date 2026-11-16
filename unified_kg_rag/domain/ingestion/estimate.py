# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cost estimate of an ingestion run, before any model call.

``run-ingestion --estimate`` parses and chunks the corpus (no model call)
and prices the model calls that its text units decide: the intelligent
chunker's boundary calls, the translation, the graph extraction, the
gleaning rounds and the claim extraction, each call at its prompt's
templates plus the text it reads, by the local token estimate the usage
meter counts a call's start at. The stages whose calls depend on the graph
the extraction finds (the description summaries, the community reports,
the summary tree) are named, not estimated.

Output tokens are not known before the calls: they are estimated at
:data:`OUTPUT_TOKENS_PER_INPUT_TOKEN` of the input, the ratio of the
measured benchmark build (docs/concepts/defaults-and-evidence.md, *How the
measurements were run*: 26.9M output and 51.4M input tokens over 3,318 short
paragraphs). The USD figure prices both at ``models.prices``
(``domain.pricing``); a model without a price leaves it out.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, Field

from unified_kg_rag.domain.models import (
    ChunkingStrategy,
    Config,
    ModelRole,
    PipelineStageType,
)
from unified_kg_rag.domain.pricing import (
    ModelPrice,
    price_of,
    token_rates,
    tokens_cost_usd,
)

__all__ = [
    "GRAPH_DEPENDENT_STAGES",
    "EstimateBound",
    "OUTPUT_TOKENS_PER_INPUT_TOKEN",
    "RunEstimate",
    "StageCalls",
    "StageEstimate",
    "STAGE_MODEL_ROLES",
    "graph_dependent_calls",
    "price_stages",
    "run_model_roles",
    "unpriced_run_models",
]

EstimateBound = Literal["exact", "at most", "about"]

# Output tokens per input token of the measured benchmark build (26.9M of
# output for 51.4M of input); an estimate's assumption, not a property of
# any model.
OUTPUT_TOKENS_PER_INPUT_TOKEN = 26.9 / 51.4


# Whether a config's run summarizes descriptions over the threshold: the
# resolution's merges, the community detection's stored members and the
# indexing's merges into the stored graph each may.
def _summarizes(config: Config) -> bool:
    return config.processing.graph_extraction.description_summarization.enabled


# The stages whose model calls depend on the graph the extraction finds: what
# each calls the model for, and whether a config's run calls it at all.
GRAPH_DEPENDENT_STAGES: dict[str, tuple[str, Callable[[Config], bool]]] = {
    "graph_resolution": (
        "a summary of each entity or relationship whose merged descriptions "
        "exceed the summary threshold",
        _summarizes,
    ),
    "community_detection": (
        "a report of each community",
        lambda c: c.graph.community_detection.enabled
        and c.graph.community_detection.report_generation.enabled,
    ),
    "summary_tree": ("a summary of each cluster of the summary tree", lambda c: True),
}


_SUMMARY = (ModelRole.DESCRIPTION_SUMMARY, _summarizes)

# The model roles each stage calls, each with whether a config's run calls
# it at all: the models a run's budget (processing.max_run_usd) must price.
# Every role a stage's code calls is listed (tested against the stage
# classes); the output fixer's is added to any run that calls a model.
STAGE_MODEL_ROLES: dict[
    PipelineStageType, tuple[tuple[ModelRole, Callable[[Config], bool]], ...]
] = {
    PipelineStageType.TEXT_CHUNKING: (
        (
            ModelRole.CHUNKING,
            lambda c: c.processing.chunking.chunker_type
            is ChunkingStrategy.INTELLIGENT,
        ),
    ),
    PipelineStageType.TRANSLATION: (
        (
            ModelRole.TRANSLATION,
            lambda c: c.processing.translation.enabled
            and not c.processing.translation.is_noop,
        ),
    ),
    PipelineStageType.GRAPH_EXTRACTION: ((ModelRole.GRAPH_EXTRACTION, lambda c: True),),
    PipelineStageType.GLEANING: (
        (ModelRole.GLEANING, lambda c: c.processing.gleaning.enabled),
    ),
    PipelineStageType.GRAPH_RESOLUTION: (_SUMMARY,),
    PipelineStageType.CLAIM_EXTRACTION: (
        (ModelRole.CLAIM_EXTRACTION, lambda c: c.processing.claim_extraction.enabled),
    ),
    PipelineStageType.COMMUNITY_DETECTION: (
        (ModelRole.COMMUNITY_REPORT, GRAPH_DEPENDENT_STAGES["community_detection"][1]),
        _SUMMARY,
    ),
    PipelineStageType.INDEXING: (_SUMMARY,),
    PipelineStageType.SUMMARY_TREE: (
        (ModelRole.SUMMARY_TREE, GRAPH_DEPENDENT_STAGES["summary_tree"][1]),
    ),
}


def run_model_roles(
    config: Config, stages: Collection[PipelineStageType]
) -> list[ModelRole]:
    """The model roles a run of ``stages`` under ``config`` calls
    (:data:`STAGE_MODEL_ROLES`), and the output fixer's when it calls any
    (it repairs their malformed output)."""
    roles = [
        role
        for stage, called in STAGE_MODEL_ROLES.items()
        if stage in stages
        for role, calls in called
        if calls(config)
    ]
    return [*dict.fromkeys(roles), ModelRole.FIXING] if roles else []


def unpriced_run_models(
    config: Config, stages: Collection[PipelineStageType]
) -> dict[str, list[str]]:
    """The models a run of ``stages`` calls that ``models.prices`` gives no
    token rates, each with the roles it serves: a run under
    ``processing.max_run_usd`` would stop at its first call."""
    unpriced: dict[str, list[str]] = {}
    for role in run_model_roles(config, stages):
        model = str(config.model_id(role))
        if token_rates(price_of(config.models.prices, model)) is None:
            unpriced.setdefault(model, []).append(role.value)
    return unpriced


def graph_dependent_calls(
    config: Config, stages: Collection[PipelineStageType]
) -> dict[str, str]:
    """Of ``stages`` (a run's), those that call a model for what the
    extraction finds under ``config``, and what for."""
    return {
        name: what
        for name, (what, calls) in GRAPH_DEPENDENT_STAGES.items()
        if PipelineStageType(name) in stages and calls(config)
    }


class StageCalls(BaseModel):
    """The model calls one stage makes for the corpus, counted from it."""

    stage: str = Field(description="The pipeline stage")
    role: str = Field(description="The model role its calls run on")
    calls: int = Field(ge=0, description="Its model calls")
    input_tokens: int = Field(ge=0, description="Their estimated input tokens")
    bound: EstimateBound = Field(
        description="exact; at most (a stage that may make fewer: gleaning "
        "rounds past the first, units a stored translation serves); or about "
        "(over the units of an intelligent chunking, cut here as without "
        "boundary answers, or over text translated into another index "
        "language, counted at the source text)"
    )


class StageEstimate(StageCalls):
    """A stage's calls, priced."""

    model: str = Field(description="The model of its role")
    output_tokens: int = Field(
        ge=0, description="Estimated output tokens (OUTPUT_TOKENS_PER_INPUT_TOKEN)"
    )
    usd: float | None = Field(description="Estimated USD; None without a price")


class RunEstimate(BaseModel):
    """What an ingestion run of a corpus would cost, before any model call."""

    documents: int = Field(ge=0, description="The parsed documents")
    text_units: int = Field(ge=0, description="The text units chunking cut")
    text_unit_tokens: int = Field(ge=0, description="Their estimated tokens")
    stages: list[StageEstimate] = Field(description="The estimated stages")
    not_estimated: dict[str, str] = Field(
        description="The stages of the run whose calls depend on the extracted "
        "graph, and what each calls the model for"
    )
    calls: int = Field(ge=0, description="The estimated stages' calls")
    input_tokens: int = Field(ge=0, description="Their input tokens")
    output_tokens: int = Field(ge=0, description="Their output tokens")
    usd: float | None = Field(description="Their USD; None when a model has no price")
    unpriced_models: list[str] = Field(description="The models without a price")
    output_tokens_per_input_token: float = Field(
        description="The output estimate's ratio (OUTPUT_TOKENS_PER_INPUT_TOKEN)"
    )


def price_stages(
    stages: Sequence[StageCalls],
    *,
    model_of: Callable[[str], str],
    prices: Mapping[str, ModelPrice],
    documents: int,
    unit_tokens: Sequence[int],
    not_estimated: Mapping[str, str] | None = None,
) -> RunEstimate:
    """The estimate of a run that makes ``stages``' calls: each priced on
    its role's model (``model_of``) at ``prices``."""
    priced: list[StageEstimate] = []
    unpriced: set[str] = set()
    for stage in stages:
        if not stage.calls:
            continue
        model = model_of(stage.role)
        output = round(stage.input_tokens * OUTPUT_TOKENS_PER_INPUT_TOKEN)
        price = price_of(prices, model)
        usd = None
        if price is not None and price.input_per_million_tokens is not None:
            usd = tokens_cost_usd(price, stage.input_tokens, output)
        else:
            unpriced.add(model)
        priced.append(
            StageEstimate(
                **stage.model_dump(), model=model, output_tokens=output, usd=usd
            )
        )
    return RunEstimate(
        documents=documents,
        text_units=len(unit_tokens),
        text_unit_tokens=sum(unit_tokens),
        stages=priced,
        not_estimated=dict(not_estimated or {}),
        calls=sum(s.calls for s in priced),
        input_tokens=sum(s.input_tokens for s in priced),
        output_tokens=sum(s.output_tokens for s in priced),
        usd=(
            None if unpriced else float(sum(s.usd for s in priced if s.usd is not None))
        ),
        unpriced_models=sorted(unpriced),
        output_tokens_per_input_token=OUTPUT_TOKENS_PER_INPUT_TOKEN,
    )
