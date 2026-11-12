# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Moved and removed configuration keys, with the hints they raise."""

import types
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

# Moved settings -> the key that replaces each. A config that sets one fails
# validation naming its replacement.
MOVED_KEYS: dict[str, str] = {
    "graph.analysis.centrality": "graph.visualization.centrality",
    "aws.bedrock.default_model_id": "models.default",
    "aws.bedrock.fast_model_id": "models.fast",
    "providers.llm.default_model_id": "models.default",
    "providers.llm.fast_model_id": "models.fast",
    "fixing.fixing_model_id": "models.roles.fixing",
    "processing.chunking.chunking_model_id": "models.roles.chunking",
    "processing.translation.translation_model_id": "models.roles.translation",
    "processing.graph_extraction.extraction_model_id": "models.roles.graph_extraction",
    "processing.graph_extraction.description_summarization.summary_model_id": (
        "models.roles.description_summary"
    ),
    "processing.gleaning.graph_refinement_model_id": "models.roles.gleaning",
    "processing.claim_extraction.extraction_model_id": "models.roles.claim_extraction",
    "graph.community_detection.report_generation.report_generation_model_id": (
        "models.roles.community_report"
    ),
    "search.translation_model_id": "models.roles.query_translation",
    "search.entity_extraction_model_id": "models.roles.entity_extraction",
    "search.strategy_selection_model_id": "models.roles.strategy_selection",
    "search.query_rewrite_model_id": "models.roles.query_rewrite",
    "search.answer_generation_model_id": "models.roles.answer_generation",
    "search.global_search.community_relevance_model_id": (
        "models.roles.community_relevance"
    ),
    "search.global_search.map_model_id": "models.roles.global_map",
    "search.drift_search.primer_model_id": "models.roles.drift_primer",
    "search.drift_search.action_model_id": "models.roles.drift_action",
    "evaluation.evaluation_model_id": "models.roles.evaluation",
    "providers.embedding.model_id": "models.embedding.model_id",
    "providers.embedding.dimension": "models.embedding.dimension",
    "indexing.opensearch.embedding_model_id": "models.embedding.model_id",
    "indexing.opensearch.embedding_dimension": "models.embedding.dimension",
    "evaluation.embedding_model_id": "models.embedding.model_id",
    "providers.rerank.model_id": "models.rerank",
    "aws.dynamodb.enabled": "providers.doc_status.enabled",
    "search.reranking.rerank_model_id": "models.rerank",
    "aws.bedrock.default_max_output_tokens": "models.default_max_output_tokens",
    # Retries are configured in one section; each old key counts what its
    # replacement counts, so a tuned value fails the load instead of
    # silently reverting to the default.
    "processing.max_attempts": "retry.max_attempts",
    "aws.bedrock.transient_retry.max_attempts": "retry.max_attempts",
    "aws.bedrock.transient_retry.base_delay_seconds": "retry.base_delay_seconds",
    "aws.bedrock.transient_retry.max_delay_seconds": "retry.max_delay_seconds",
    "aws.bedrock.transient_retry.max_total_seconds": "retry.max_total_seconds",
    "providers.llm.max_attempts": "retry.model_server_max_attempts",
    "providers.embedding.max_attempts": "retry.model_server_max_attempts",
    "indexing.neptune.max_attempts": "retry.graph_store_max_attempts",
    "indexing.neptune.retry_delay_seconds": "retry.base_delay_seconds",
    "search.store_max_attempts": "retry.query_store_max_attempts",
    "evaluation.ragas_max_attempts": "retry.judge_max_attempts",
    # The 0.2.3 retry counts, which count otherwise (MOVED_KEY_NOTES).
    "processing.max_retries": "retry.max_attempts",
    "indexing.neptune.max_retries": "retry.graph_store_max_attempts",
    # 0.2.3's call pools, one pool of in-flight calls now.
    "processing.chunk_concurrency": "processing.max_concurrency",
    "processing.batch_size": "processing.max_concurrency",
    "indexing.cross_run_fuzzy_merge": "processing.fuzzy_entity_resolution",
    # Query-time traversal settings every graph store reads.
    **{
        f"indexing.neptune.{key}": f"search.graph_expansion.{key}"
        for key in (
            "max_hops",
            "max_results_per_hop",
            "entity_importance_source",
            "traversal_fetch_multiplier",
            "deterministic_order",
        )
    },
}

# Moved key -> what to know when converting its value (a key whose
# replacement counts the same needs none).
MOVED_KEY_NOTES: dict[str, str] = {
    "processing.max_retries": (
        "it had no effect; each model call made 5 attempts, the "
        "retry.max_attempts default"
    ),
    "indexing.neptune.max_retries": (
        "it counted retries, so max_retries N is graph_store_max_attempts N+1"
    ),
    "processing.batch_size": (
        "it counted the items of one batch, not calls in flight; "
        "max_concurrency caps the calls"
    ),
}


# Removed settings -> why they are gone. A config that sets one, or a key
# under a removed section, loads with a WARNING and the setting ignored.
REMOVED_KEYS: dict[str, str] = {
    "aws.bedrock.guardrail.apply_to": (
        "the guardrail screens the user query and the answer only; ingestion "
        "and evaluation models are never guarded"
    ),
    "indexing.cross_run_merge": (
        "an incremental run always merges its delta into the stored graph; "
        "without the merge (false, 0.2.3's default) a run overwrote entities "
        "other documents still cite"
    ),
    **{
        f"graph.visualization.{key}": (
            "ingestion no longer renders visualizations; run-visualization "
            "renders the visualization_data.json it writes "
            "(graph.visualization.export_data), with a spring layout"
        )
        for key in (
            "enabled",
            "embedding_method",
            "layout_method",
            "layout",
            "embeddings",
        )
    },
    **{
        f"indexing.opensearch.{key}": (
            "the persisted S3 embedding cache is gone; an incremental run "
            "(providers.doc_status.enabled) embeds only the documents it "
            "re-indexes"
        )
        for key in ("persist_embedding_cache", "embedding_cache_s3_key")
    },
    "cache.chunking": (
        "a stage-cache entry of more than 1,000 items or 50 MB is always "
        "written in chunk files"
    ),
    "search.lightrag_search.structured_keyword_output": (
        "keyword extraction parses the model's text answer; the structured "
        "output pilot was removed before any release"
    ),
    **dict.fromkeys(
        (
            "search.global_search.reduce_with_llm",
            "search.global_search.map_reduce_model_id",
            "models.roles.global_reduce",
        ),
        "global search runs no reduce call: the ranked key points go to the "
        "answer model as one context section (0.2.3 reduced them with "
        "map_reduce_model_id by default)",
    ),
    "search.fusion.method": (
        "results are always fused by reciprocal rank fusion (rrf, the default "
        "before); weighted fusion is gone"
    ),
    **{
        f"graph.visualization.centrality.{key}": (
            "the visualization computes degree, betweenness and PageRank "
            "centrality; closeness and eigenvector centrality were never "
            "displayed"
        )
        for key in (
            "calculate_closeness",
            "calculate_eigenvector",
            "eigenvector_max_iter",
            "eigenvector_tol",
        )
    },
    "search.reranking.enabled": (
        "providers.rerank.kind picks the reranker, so reranking stays on; set "
        "providers.rerank.kind: none to turn it off"
    ),
    **{
        f"graph.analysis.statistics.{key}": (
            "the graph_analysis stage computes the density and the connected "
            "components; no stage read the average clustering coefficient or "
            "the diameter, which are no longer computed"
        )
        for key in (
            "calculate_density",
            "calculate_clustering",
            "calculate_components",
            "calculate_average_clustering",
            "calculate_diameter",
        )
    },
    "fixing": (
        "output fixing is always on (the default before); models.roles.fixing "
        "picks its model"
    ),
    "evaluation.warm_up_query": (
        "run-eval always answers one untimed, unscored warm-up query per "
        "strategy and run (the default before)"
    ),
    **dict.fromkeys(
        (
            "graph.community_detection.auto_resolution",
            "graph.community_detection.auto_resolution_candidates",
            "graph.community_detection.auto_resolution_max_nodes",
        ),
        "the modularity sweep is gone (it merged unrelated communities on the "
        "upper levels); set graph.community_detection.resolution",
    ),
    "aws.dynamodb.billing_mode": (
        "a table the library creates is on-demand (PAY_PER_REQUEST, the only "
        "value that worked); provision the table yourself, with "
        "aws.dynamodb.create_table_if_missing: false, for another mode"
    ),
    "evaluation.langchain_metrics": (
        "the langchain evaluator is removed; ragas' answer_correctness "
        "(evaluation.ragas_metrics) scores correctness against the answer"
    ),
}


def _sets(data: Any, path: str) -> bool:
    node = data
    for part in path.split("."):
        node = node.get(part, _ABSENT) if isinstance(node, dict) else _ABSENT
    return node is not _ABSENT


def moved_key_hint(old: str, new: str) -> str:
    """``new``, with what to know when converting ``old``'s value, if any."""
    note = MOVED_KEY_NOTES.get(old)
    return f"{new} ({note})" if note else new


def moved_keys(data: Any) -> list[str]:
    """``"old -> new"`` for each moved setting ``data`` sets."""
    return [
        f"{old} -> {moved_key_hint(old, new)}"
        for old, new in MOVED_KEYS.items()
        if _sets(data, old)
    ]


def removed_keys(data: Any) -> list[str]:
    """The removed settings ``data`` sets (see ``REMOVED_KEYS``)."""
    return [path for path in REMOVED_KEYS if _sets(data, path)]


_ABSENT: Any = object()


def nested_model(annotation: Any) -> type[BaseModel] | None:
    """The BaseModel behind ``Model`` / ``Model | None`` (not dict/list values)."""
    if get_origin(annotation) not in (None, Union, types.UnionType):
        return None
    for candidate in get_args(annotation) or (annotation,):
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def unknown_keys(data: Any, model: type[BaseModel], prefix: str = "") -> list[str]:
    """The config keys in ``data`` the model does not define (pydantic would
    drop them), as dotted paths. Moved keys are left out (validation refuses
    them, naming their replacement), and so are removed ones (ignored with a
    warning)."""
    if not isinstance(data, dict):
        return []
    found: list[str] = []
    for key, value in data.items():
        path = f"{prefix}{key}"
        field = model.model_fields.get(key)
        if field is None:
            if path not in MOVED_KEYS and path not in REMOVED_KEYS:
                found.extend(_unknown_in_gone_section(path, value))
            continue
        nested = nested_model(field.annotation)
        if nested is not None:
            found.extend(unknown_keys(value, nested, f"{path}."))
    return found


def _unknown_in_gone_section(path: str, value: Any) -> list[str]:
    """The unknown keys at ``path``, a key no model defines: none of a
    section that only holds moved or removed keys (``graph.analysis``,
    whose last settings went) but the keys in it that are neither, else
    ``path`` itself."""
    gone = [*MOVED_KEYS, *REMOVED_KEYS]
    if not isinstance(value, dict) or not any(k.startswith(f"{path}.") for k in gone):
        return [path]
    found: list[str] = []
    for key, inner in value.items():
        inner_path = f"{path}.{key}"
        if inner_path not in MOVED_KEYS and inner_path not in REMOVED_KEYS:
            found.extend(_unknown_in_gone_section(inner_path, inner))
    return found


def without_keys(data: dict[str, Any], paths: list[str]) -> dict[str, Any]:
    """``data`` without the dotted ``paths`` (``unknown_keys``), copied
    where it changes: what ``allow_unknown_keys`` validates."""
    stripped = dict(data)
    for path in paths:
        *parents, last = path.split(".")
        node = stripped
        for key in parents:
            child = node.get(key)
            if not isinstance(child, dict):
                break
            node[key] = node = dict(child)
        else:
            node.pop(last, None)
    return stripped
