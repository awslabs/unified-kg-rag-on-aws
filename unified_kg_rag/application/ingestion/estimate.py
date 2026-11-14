# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``run-ingestion --estimate``: a corpus's model calls, tokens and USD
before a run (``domain.ingestion.estimate``).

The corpus is parsed and chunked as a full build of it would (the stages'
own parser and chunkers, no model call: an intelligent chunking is cut as
its chunker cuts a pre-chunk no boundary answer came for, ``PreChunker``),
and each stage whose calls the text units decide is counted: one call per
unit for the extraction, the claims and each translation language, one per
gleaning round and unit (``max_rounds``: at most, as a round re-sends only
the units the last one added to), one per pre-chunk the intelligent chunker
asks about (the claims and the translation at most too: a unit without an
entity is asked for no claim, one a stored translation serves is not
translated). A call's input is its prompt's templates and the type blocks
the config fills in, plus the text it reads (a gleaning call the unit's
extraction too, at its estimated output), by the token estimate the usage
meter counts a call's start at. No store or model is reached.
"""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from pathlib import Path

from unified_kg_rag.adapters.ingestion.chunker import PreChunker, SimpleTextChunker
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.application.ingestion.pipeline_stages import DocumentParsingStage
from unified_kg_rag.domain.ingestion.delta_detector import assign_document_identity
from unified_kg_rag.domain.ingestion.estimate import (
    OUTPUT_TOKENS_PER_INPUT_TOKEN,
    EstimateBound,
    RunEstimate,
    StageCalls,
    graph_dependent_calls,
    price_stages,
)
from unified_kg_rag.domain.ingestion.methodologies import selected_stages
from unified_kg_rag.domain.ingestion.seed_graph import is_seed_document
from unified_kg_rag.domain.models import (
    Config,
    Document,
    ModelRole,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
)
from unified_kg_rag.domain.models.config import ChunkingStrategy
from unified_kg_rag.domain.prompts import (
    BasePrompt,
    ClaimExtractionPrompt,
    GraphExtractionPrompt,
    GraphRefinementPrompt,
    TextChunkingPrompt,
    TextTranslationPrompt,
)

__all__ = ["estimate_ingestion"]

_S = PipelineStageType


def _prompt_tokens(prompt: type[BasePrompt], config: Config) -> int:
    """The tokens of ``prompt``'s templates and the blocks ``config`` fills
    into them: what each of its calls reads besides its own text."""
    resolved = prompt.resolve(custom_prompts=config.custom_prompts)
    filled = getattr(prompt, "config_variables", None)
    blocks = "".join(filled(config).values()) if filled is not None else ""
    return estimate_token_count(
        resolved.system_prompt_template + resolved.human_prompt_template + blocks
    )


def _parse(config: Config, source: Path) -> list[Document]:
    """The corpus's documents, parsed by the parsing stage (no registry:
    the estimate is a full build's) with their identity assigned."""
    context = PipelineContext(
        pipeline_id="estimate",
        config=PipelineConfig(),
        status=PipelineStageStatus.RUNNING,
        start_time=datetime.now(),
        source_directory=source,
    )
    result = DocumentParsingStage(config, source).execute(context)
    if result.status is PipelineStageStatus.FAILED:
        raise ValueError(result.error_message or "the corpus could not be parsed")
    documents = [d for d in context.documents if not is_seed_document(d)]
    for document in documents:
        assign_document_identity(document, source)
    return documents


def estimate_ingestion(
    config: Config,
    source_directory: str | Path,
    *,
    stages: Collection[PipelineStageType] | None = None,
) -> RunEstimate:
    """The estimate of a full build of ``source_directory`` under
    ``config`` (see the module docstring), of ``stages`` only when given
    (``run-ingestion --enabled-stages``): of them, those ``config``
    selects."""
    source = Path(source_directory).resolve()
    documents = _parse(config, source)
    chunking = config.processing.chunking
    intelligent = chunking.chunker_type is ChunkingStrategy.INTELLIGENT
    chunker = (
        PreChunker(config, show_progress=False)
        if intelligent
        else SimpleTextChunker(config, show_progress=False)
    )
    units = chunker.chunk_documents(documents)
    unit_tokens = [estimate_token_count(unit.text) for unit in units]
    selected = selected_stages(config)
    stages = [s for s in selected if s in stages] if stages is not None else selected
    processing = config.processing
    rows: list[StageCalls] = []

    def per_unit(
        stage: PipelineStageType,
        role: ModelRole,
        prompt: type[BasePrompt],
        *,
        rounds: int = 1,
        extra_per_unit: int = 0,
        bound: EstimateBound = "exact",
    ) -> None:
        overhead = _prompt_tokens(prompt, config) + extra_per_unit
        rows.append(
            StageCalls(
                stage=stage.value,
                role=role.value,
                calls=len(units) * rounds,
                input_tokens=rounds * (len(units) * overhead + sum(unit_tokens)),
                bound="about" if intelligent and bound == "exact" else bound,
            )
        )

    if intelligent and _S.TEXT_CHUNKING in stages:
        assert isinstance(chunker, PreChunker)
        asked = [
            pre
            for document in documents
            for pre in chunker.pre_chunks_of(document)
            if chunker.needs_boundary_call(pre)
        ]
        overhead = _prompt_tokens(TextChunkingPrompt, config)
        rows.append(
            StageCalls(
                stage=_S.TEXT_CHUNKING.value,
                role=ModelRole.CHUNKING.value,
                calls=len(asked),
                # The pre-chunk is sent with a line number on each line.
                input_tokens=sum(
                    overhead + estimate_token_count(pre) + pre.count("\n") + 1
                    for pre in asked
                ),
                bound="exact",
            )
        )
    translation = processing.translation
    if _S.TRANSLATION in stages and translation.enabled and not translation.is_noop:
        languages = len(
            [
                code
                for code in translation.translated_languages
                if code != translation.source_language.value
            ]
        )
        per_unit(
            _S.TRANSLATION,
            ModelRole.TRANSLATION,
            TextTranslationPrompt,
            rounds=languages,
            bound="at most",
        )
    extraction_overhead = _prompt_tokens(GraphExtractionPrompt, config)
    if _S.GRAPH_EXTRACTION in stages:
        per_unit(_S.GRAPH_EXTRACTION, ModelRole.GRAPH_EXTRACTION, GraphExtractionPrompt)
    if _S.GLEANING in stages and processing.gleaning.enabled:
        # A round reads the unit's extraction (about the extraction's
        # output) beside the text.
        extraction_output = round(
            (extraction_overhead + (sum(unit_tokens) / max(1, len(units))))
            * OUTPUT_TOKENS_PER_INPUT_TOKEN
        )
        per_unit(
            _S.GLEANING,
            ModelRole.GLEANING,
            GraphRefinementPrompt,
            rounds=processing.gleaning.max_rounds,
            extra_per_unit=extraction_output,
            bound="at most",
        )
    if _S.CLAIM_EXTRACTION in stages and processing.claim_extraction.enabled:
        # A unit whose extraction names no entity is not asked.
        per_unit(
            _S.CLAIM_EXTRACTION,
            ModelRole.CLAIM_EXTRACTION,
            ClaimExtractionPrompt,
            bound="at most",
        )
    return price_stages(
        rows,
        model_of=lambda role: config.model_id(ModelRole(role)),
        prices=config.models.prices,
        documents=len(documents),
        unit_tokens=unit_tokens,
        not_estimated=graph_dependent_calls(config, stages),
    )
