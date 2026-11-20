# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Moved, deprecated and removed configuration keys, with the hints they
raise."""

import copy
import functools
import types
from collections.abc import Iterator
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
    # Retries are configured in one section, one profile per caller; each old
    # key counts what its replacement counts, so a tuned value fails the load
    # instead of silently reverting to the default.
    "processing.max_attempts": "retry.ingestion_item.max_attempts",
    "aws.bedrock.transient_retry.max_attempts": "retry.model.max_attempts",
    "aws.bedrock.transient_retry.base_delay_seconds": "retry.model.base_delay_seconds",
    "aws.bedrock.transient_retry.max_delay_seconds": "retry.model.max_delay_seconds",
    "aws.bedrock.transient_retry.max_total_seconds": "retry.model.max_total_seconds",
    "providers.llm.max_attempts": "retry.model_server.max_attempts",
    "providers.embedding.max_attempts": "retry.model_server.max_attempts",
    "indexing.neptune.max_attempts": "retry.graph_store.max_attempts",
    "indexing.neptune.retry_delay_seconds": "retry.graph_store.base_delay_seconds",
    "search.store_max_attempts": "retry.query_store.max_attempts",
    "evaluation.ragas_max_attempts": "retry.judge_max_attempts",
    # The 0.2.3 retry counts, which count otherwise (MOVED_KEY_NOTES).
    "processing.max_retries": "retry.ingestion_item.max_attempts",
    "indexing.neptune.max_retries": "retry.graph_store.max_attempts",
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
        "retry.ingestion_item.max_attempts default"
    ),
    "indexing.neptune.max_retries": (
        "it counted retries, so max_retries N is retry.graph_store.max_attempts N+1"
    ),
    "processing.batch_size": (
        "it counted the items of one batch, not calls in flight; "
        "max_concurrency caps the calls"
    ),
}


# Deprecated settings -> the key, or keys, that replace each. A config that
# sets one still loads: its value is moved into every replacement before
# validation (:func:`migrate_deprecated_keys`), with a WARNING naming it.
DEPRECATED_KEYS: dict[str, str | tuple[str, ...]] = {
    "evaluation.prices": "models.prices",
    # The namespace's one name (domain/models/namespace.py).
    "processing.document_parsing.index_value": "processing.document_parsing.namespace",
    # The flat retry keys of 0.3.0, before the per-caller profiles: each
    # governed the callers whose profiles it maps to.
    "retry.max_attempts": (
        "retry.model.max_attempts",
        "retry.ingestion_item.max_attempts",
    ),
    "retry.base_delay_seconds": (
        "retry.model.base_delay_seconds",
        "retry.model_server.base_delay_seconds",
        "retry.graph_store.base_delay_seconds",
    ),
    "retry.max_delay_seconds": (
        "retry.model.max_delay_seconds",
        "retry.model_server.max_delay_seconds",
    ),
    "retry.max_total_seconds": (
        "retry.model.max_total_seconds",
        "retry.model_server.max_total_seconds",
    ),
    "retry.model_server_max_attempts": "retry.model_server.max_attempts",
    "retry.graph_store_max_attempts": "retry.graph_store.max_attempts",
    "retry.query_store_max_attempts": "retry.query_store.max_attempts",
}


def deprecated_targets(old: str) -> tuple[str, ...]:
    """The keys the deprecated ``old`` loads into."""
    new = DEPRECATED_KEYS[old]
    return (new,) if isinstance(new, str) else new


def deprecated_key_warning(old: str) -> str:
    """The warning a config that sets the deprecated ``old`` gets: the
    loader's log line and the console's settings check say the same."""
    targets = deprecated_targets(old)
    if len(targets) == 1:
        return f"Config key {old} is deprecated: it loads into {targets[0]}; set that key instead"
    return (
        f"Config key {old} is deprecated: it governed several callers and loads "
        f"into each of {', '.join(targets)}; set those keys instead"
    )


def removed_key_warning(path: str, where: str | None = None) -> str:
    """The warning a config that sets the removed ``path`` gets (``where``:
    how the loader names it, a file key or an environment variable)."""
    return f"Config key {where or path} is removed and ignored: {REMOVED_KEYS[path]}"


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
    return _get(data, path) is not _ABSENT


def migrate_deprecated_keys(data: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """``data`` with every deprecated key it sets moved into each key that
    replaces it (:func:`deprecated_targets`; copied where it changes), and
    those deprecated keys. A replacement ``data`` sets too keeps its value: a
    mapping (``models.prices``) keeps its own entries, whole, over the
    deprecated key's, any other value set otherwise disagrees.

    Raises:
        ValueError: A deprecated key and a replacement are both set, to
            different values (naming both).
    """
    data, migrated, conflicts = _migrate(data)
    if conflicts:
        raise ValueError(
            "deprecated settings disagree with their replacements: "
            + "; ".join(
                f"{old} and {new} are both set, to different values"
                for old, new in conflicts
            )
            + " (set the replacement only)"
        )
    return data, migrated


def deprecated_keys(data: dict[str, Any]) -> dict[str, list[str]]:
    """The deprecated keys ``data`` sets, each with the replacements it also
    sets to a different value (the disagreement
    :func:`migrate_deprecated_keys` refuses; none when they agree)."""
    _, migrated, conflicts = _migrate(data)
    return {
        old: [new for conflicting, new in conflicts if conflicting == old]
        for old in migrated
    }


def _migrate(
    data: dict[str, Any],
) -> tuple[dict[str, Any], list[str], list[tuple[str, str]]]:
    """``data`` migrated, the deprecated keys moved and the
    ``(deprecated, replacement)`` pairs set to different values."""
    migrated: list[str] = []
    conflicts: list[tuple[str, str]] = []
    for old in DEPRECATED_KEYS:
        if not _sets(data, old):
            continue
        if not migrated:
            data = copy.deepcopy(data)
        migrated.append(old)
        conflicts.extend((old, new) for new in _move(data, old))
    return data, migrated, conflicts


def retire_deprecated_key(data: dict[str, Any], old: str) -> dict[str, Any]:
    """``data`` (copied) without the deprecated ``old``, its value moved
    into each replacement ``data`` does not set (a mapping merged under the
    replacement's own entries, as :func:`migrate_deprecated_keys` loads
    it); a replacement set to another value keeps it, as the refusal of the
    disagreement asks. ``data`` unchanged when it does not set ``old``."""
    if not _sets(data, old):
        return data
    data = copy.deepcopy(data)
    _move(data, old)
    return data


def _move(data: dict[str, Any], old: str) -> list[str]:
    """Move the deprecated ``old`` (which ``data`` sets) into its
    replacements, in place: the replacements set to a different value,
    which keep theirs."""
    value = _pop(data, old)
    disagree = []
    for new in deprecated_targets(old):
        current = _get(data, new)
        if current is _ABSENT:
            _put(data, new, copy.deepcopy(value))
        elif isinstance(current, dict) and isinstance(value, dict):
            _put(data, new, {**value, **current})
        elif current != value:
            disagree.append(new)
    return disagree


def _get(data: Any, path: str) -> Any:
    """The value ``data`` sets at ``path``: a section given as a model (a
    ``RetryPolicy`` object) sets the fields it was given."""
    node = data
    for part in path.split("."):
        if isinstance(node, BaseModel):
            node = getattr(node, part) if part in node.model_fields_set else _ABSENT
        else:
            node = node.get(part, _ABSENT) if isinstance(node, dict) else _ABSENT
    return node


def _pop(data: dict[str, Any], path: str) -> Any:
    *parents, last = path.split(".")
    node = data
    for part in parents:
        node = node[part]
    return node.pop(last)


def _put(data: dict[str, Any], path: str, value: Any) -> None:
    *parents, last = path.split(".")
    node = data
    for part in parents:
        child = node.get(part)
        if isinstance(child, BaseModel):
            # Its set fields, so the value joins them.
            child = node[part] = child.model_dump(exclude_unset=True)
        elif not isinstance(child, dict):
            child = node[part] = {}
        node = child
    node[last] = value


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

# The keys no section defines that are not unknown: moved (refused, naming
# the replacement), deprecated (loaded into it) and removed (ignored).
_GONE = frozenset({*MOVED_KEYS, *DEPRECATED_KEYS, *REMOVED_KEYS})


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
    them, naming their replacement), and so are deprecated ones (loaded into
    their replacement) and removed ones (ignored with a warning)."""
    if not isinstance(data, dict):
        return []
    found: list[str] = []
    for key, value in data.items():
        path = f"{prefix}{key}"
        field = model.model_fields.get(key)
        if field is None:
            if path not in _GONE:
                found.extend(_unknown_in_gone_section(path, value))
            continue
        nested = nested_model(field.annotation)
        if nested is not None:
            found.extend(unknown_keys(value, nested, f"{path}."))
    return found


def _sections(
    model: type[BaseModel], prefix: str = ""
) -> Iterator[tuple[str, type[BaseModel]]]:
    """Each section of ``model``'s tree, as its dotted path and its type."""
    for key, field in model.model_fields.items():
        nested = nested_model(field.annotation)
        if nested is not None:
            yield f"{prefix}{key}", nested
            yield from _sections(nested, f"{prefix}{key}.")


@functools.cache
def _section_tree(root: type[BaseModel]) -> tuple[tuple[str, type[BaseModel]], ...]:
    return tuple(_sections(root))


def _sections_taking(
    model: type[BaseModel], key: str, root: type[BaseModel]
) -> tuple[str, ...]:
    """The sections of ``root`` that take ``key`` where a section of type
    ``model`` does not: those of a subclass of it that defines the key
    (``retry.bulk_item`` and ``retry.model``, of type ``ClientRetryPolicy``,
    for a ``RetryPolicy``'s ``client_max_attempts``)."""
    if key in model.model_fields:
        return ()
    return tuple(
        sorted(
            path
            for path, section in _section_tree(root)
            if section is not model
            and issubclass(section, model)
            and key in section.model_fields
        )
    )


def _taken_by_note(sections: tuple[str, ...]) -> str:
    """What a refusal of a key adds when other ``sections`` take it."""
    if not sections:
        return ""
    names = ", ".join(sections[:-1]) + " and " if len(sections) > 1 else ""
    return f" (only {names}{sections[-1]} take it)"


@functools.cache
def subclass_keys(root: type[BaseModel]) -> dict[str, tuple[str, ...]]:
    """The keys a section of ``root`` refuses that a section of a subclass
    of its type takes, as dotted paths, each with the sections that take it
    (:func:`_sections_taking`): what an unknown-key refusal names, whether
    the key comes from a file, a variable or a constructor."""
    found: dict[str, tuple[str, ...]] = {}
    for path, model in _section_tree(root):
        for _, section in _section_tree(root):
            if section is not model and issubclass(section, model):
                for key in section.model_fields.keys() - model.model_fields.keys():
                    found[f"{path}.{key}"] = _sections_taking(model, key, root)
    return found


def unknown_key_note(path: str, root: type[BaseModel]) -> str:
    """What a refusal of the unknown key ``path`` adds: the sections that
    take it, should a section of a subclass of its section's type
    (:func:`subclass_keys`)."""
    return _taken_by_note(subclass_keys(root).get(path, ()))


def _defined_keys_only(data: Any, model: type[BaseModel]) -> Any:
    """``data`` without the keys the model does not define, copied where it
    changes: once the unknown and moved keys are refused, what the removed
    settings and the sections they emptied (``graph.analysis``) leave out
    of validation."""
    if not isinstance(data, dict):
        return data
    kept: dict[str, Any] = {}
    for key, value in data.items():
        field = model.model_fields.get(key)
        if field is None:
            continue
        nested = nested_model(field.annotation)
        kept[key] = value if nested is None else _defined_keys_only(value, nested)
    return kept


def _unknown_in_gone_section(path: str, value: Any) -> list[str]:
    """The unknown keys at ``path``, a key no model defines: none of a
    section that only holds moved or removed keys (``graph.analysis``,
    whose last settings went) but the keys in it that are neither, else
    ``path`` itself."""
    if not isinstance(value, dict) or not any(k.startswith(f"{path}.") for k in _GONE):
        return [path]
    found: list[str] = []
    for key, inner in value.items():
        inner_path = f"{path}.{key}"
        if inner_path not in _GONE:
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
