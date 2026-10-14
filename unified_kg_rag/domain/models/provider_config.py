# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-role provider selection (``backend`` / ``providers`` config sections).

Each role — language model, embedding model, reranker, graph store, vector
store, document-status registry — names the provider ``kind`` that serves it.
The defaults are the AWS services (Bedrock, Neptune, OpenSearch, DynamoDB), so
a config without these sections behaves exactly as before. ``backend: local``
is a preset that fills in local providers (Ollama, a cross-encoder reranker
and the embedded SQLite graph, vector and registry stores) for every key the
config does not set itself.

A ``kind`` is a plain string resolved at runtime through the provider registry
(``unified_kg_rag.adapters.provider_registry``), so a custom kind can be
registered without editing this module. The built-in kinds are listed on each
field.
"""

from __future__ import annotations

import copy
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, SecretStr, StringConstraints

ProviderKind = Annotated[
    str, StringConstraints(strip_whitespace=True, to_lower=True, min_length=1)
]

# Built-in kinds. Plain strings rather than an Enum so the registry can accept
# custom kinds; these names are what the shipped adapters register under.
BEDROCK = "bedrock"
OPENAI_COMPATIBLE = "openai_compatible"
OLLAMA = "ollama"
CROSS_ENCODER = "cross_encoder"
NO_RERANK = "none"
NEPTUNE = "neptune"
GREMLIN = "gremlin"
OPENSEARCH = "opensearch"
DYNAMODB = "dynamodb"
SQLITE = "sqlite"


class Backend(str, Enum):
    """Preset that fills in the ``providers`` section (see ``LOCAL_PRESET``)."""

    AWS = "aws"
    LOCAL = "local"


class EndpointConfig(BaseModel):
    base_url: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Endpoint of the model server, e.g. http://localhost:11434 for "
            "Ollama or http://localhost:8000/v1 for an OpenAI-compatible server "
            "(vLLM, llama.cpp, LM Studio). Ignored by bedrock."
        ),
    )
    api_key: SecretStr | None = Field(
        default=None,
        description=(
            "API key for an OpenAI-compatible server (masked in logs). Most "
            "local servers ignore it; null sends a placeholder. Ignored by "
            "bedrock and ollama."
        ),
    )
    timeout_seconds: float = Field(
        default=300.0,
        gt=0,
        description="HTTP timeout per request to the model server, in seconds.",
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            "Attempts per request on transient errors, including the first. "
            "The OpenAI client retries itself; on Ollama, whose client does not, "
            "query chains and embedding calls are retried by the framework "
            "(ingestion chains keep processing.max_attempts). 1 disables the "
            "retry."
        ),
    )


class LLMProviderConfig(EndpointConfig):
    kind: ProviderKind = Field(
        default=BEDROCK,
        description="Language-model provider: bedrock | openai_compatible | ollama.",
    )
    default_model_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Model for every role on the 'default' tier when kind is not "
            "bedrock (replaces aws.bedrock.default_model_id). Required then; a "
            "role's own *_model_id still overrides it."
        ),
    )
    fast_model_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Model for every role on the 'fast' tier when kind is not bedrock "
            "(replaces aws.bedrock.fast_model_id). null uses default_model_id."
        ),
    )
    context_window_size: int = Field(
        default=32768,
        ge=1024,
        description=(
            "Context window of the non-Bedrock models, in tokens. Sizes the "
            "retrieval context budget and is sent to Ollama as num_ctx. Bedrock "
            "models take theirs from the capability catalog."
        ),
    )
    max_output_tokens: int = Field(
        default=8192,
        ge=1,
        description=(
            "Output-token ceiling of the non-Bedrock models. A request asks for "
            "aws.bedrock.default_max_output_tokens (or a prompt's own floor), "
            "clamped to this."
        ),
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the non-Bedrock models.",
    )
    tokenizer: Literal["estimate", "tiktoken"] = Field(
        default="estimate",
        description=(
            "Token counting for the non-Bedrock models: 'estimate' (the "
            "script-aware estimate Bedrock also falls back to) or 'tiktoken' "
            "(needs the [local] extra; tiktoken downloads its encoding file on "
            "first use). Falls back to the estimate when tiktoken is unavailable."
        ),
    )
    tiktoken_encoding: str = Field(
        default="cl100k_base",
        min_length=1,
        description="tiktoken encoding name used when tokenizer is 'tiktoken'.",
    )


class EmbeddingProviderConfig(EndpointConfig):
    kind: ProviderKind = Field(
        default=BEDROCK,
        description="Embedding provider: bedrock | openai_compatible | ollama.",
    )
    model_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Embedding model when kind is not bedrock. Required then; it "
            "becomes the default of indexing.opensearch.embedding_model_id and "
            "evaluation.embedding_model_id."
        ),
    )
    dimension: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Vector dimension of model_id (fixed into the OpenSearch knn_vector "
            "mappings). Required when kind is not bedrock."
        ),
    )
    max_input_tokens: int = Field(
        default=8192,
        ge=1,
        description=(
            "Longest input the non-Bedrock embedding model accepts, in tokens; "
            "longer texts are truncated (by the estimate) before embedding."
        ),
    )


class RerankProviderConfig(BaseModel):
    kind: ProviderKind = Field(
        default=BEDROCK,
        description=(
            "Reranker: bedrock | cross_encoder (sentence-transformers, [local] "
            "extra) | none (keep the fused order). With none, or a cross-encoder "
            "that cannot load, queries run unreranked, logged once and visible "
            "as kg_rag.rerank_enabled=false on the fusion span."
        ),
    )
    model_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Cross-encoder model when kind is cross_encoder. Required then; it "
            "becomes the default of search.reranking.rerank_model_id."
        ),
    )
    device: str | None = Field(
        default=None,
        description="Torch device for the cross-encoder (null lets it choose).",
    )
    max_length: int = Field(
        default=512,
        ge=16,
        description="Longest query + passage pair the cross-encoder scores, in tokens.",
    )
    batch_size: int = Field(
        default=16, ge=1, description="Pairs scored per cross-encoder batch."
    )


class GraphProviderConfig(BaseModel):
    kind: ProviderKind = Field(
        default=NEPTUNE,
        description=(
            "Graph store: neptune | gremlin (a TinkerPop Gremlin Server through "
            "the same adapter, configured by aws.neptune with use_iam: false) | "
            "sqlite (an embedded single-host graph in providers.graph.sqlite_path)."
        ),
    )
    sqlite_path: str = Field(
        default=".unified-kg-rag/graph.sqlite3",
        min_length=1,
        description="SQLite database file when kind is sqlite (created on first "
        "use). A relative path is resolved against the config file's directory "
        "(the working directory without a config file).",
    )


class VectorProviderConfig(BaseModel):
    kind: ProviderKind = Field(
        default=OPENSEARCH,
        description=(
            "Vector/lexical store: opensearch (Amazon OpenSearch Service or a "
            "local OpenSearch, configured by aws.opensearch) | sqlite (an "
            "embedded store in sqlite_path that answers the same queries, for "
            "about 100k documents per index; no server)."
        ),
    )
    sqlite_path: str = Field(
        default=".unified-kg-rag/vector.sqlite3",
        min_length=1,
        description="SQLite database file when kind is sqlite (created on first "
        "use). A relative path is resolved against the config file's directory "
        "(the working directory without a config file).",
    )


class DocStatusProviderConfig(BaseModel):
    kind: ProviderKind = Field(
        default=DYNAMODB,
        description="Document-status registry for incremental indexing: dynamodb | sqlite.",
    )
    enabled: bool | None = Field(
        default=None,
        description=(
            "Turn incremental indexing (the registry) on or off. null follows "
            "aws.dynamodb.enabled, so existing configs keep their behaviour."
        ),
    )
    sqlite_path: str = Field(
        default=".unified-kg-rag/doc-status.sqlite3",
        min_length=1,
        description="SQLite database file when kind is sqlite (created on first "
        "use). A relative path is resolved against the config file's directory "
        "(the working directory without a config file).",
    )


class ProvidersConfig(BaseModel):
    llm: LLMProviderConfig = Field(
        default_factory=LLMProviderConfig, description="Language-model provider"
    )
    embedding: EmbeddingProviderConfig = Field(
        default_factory=EmbeddingProviderConfig, description="Embedding provider"
    )
    rerank: RerankProviderConfig = Field(
        default_factory=RerankProviderConfig, description="Reranking provider"
    )
    graph: GraphProviderConfig = Field(
        default_factory=GraphProviderConfig, description="Graph store"
    )
    vector: VectorProviderConfig = Field(
        default_factory=VectorProviderConfig, description="Vector/lexical store"
    )
    doc_status: DocStatusProviderConfig = Field(
        default_factory=DocStatusProviderConfig,
        description="Document-status registry",
    )


_OLLAMA_URL = "http://localhost:11434"
LOCAL_MAX_CONTEXT_TOKENS = 22_000

# What `backend: local` fills in, under whatever the config sets itself. The
# stores are the embedded SQLite ones (no server); the aws.neptune and
# aws.opensearch entries point the gremlin and opensearch kinds at
# docker/compose.local.yaml when a config selects them. The models are small
# multilingual Ollama models (the user pulls them; nothing here downloads
# anything).
LOCAL_PRESET: dict[str, Any] = {
    "providers": {
        "llm": {
            "kind": OLLAMA,
            "base_url": _OLLAMA_URL,
            "default_model_id": "qwen2.5:7b-instruct",
        },
        "embedding": {
            "kind": OLLAMA,
            "base_url": _OLLAMA_URL,
            "model_id": "bge-m3",
            "dimension": 1024,
        },
        "rerank": {
            "kind": CROSS_ENCODER,
            "model_id": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        },
        "graph": {"kind": SQLITE},
        "vector": {"kind": SQLITE},
        "doc_status": {"kind": SQLITE},
    },
    "aws": {
        "neptune": {
            "endpoint": "localhost",
            "port": 8182,
            "use_ssl": False,
            "use_iam": False,
        },
        "opensearch": {
            "endpoint": "localhost",
            "port": 9200,
            "use_ssl": False,
            "verify_certs": False,
            "allow_anonymous": True,
        },
    },
    # The retrieval context budget the default local model accepts: its
    # 32,768-token window (providers.llm.context_window_size) minus the
    # 8,192-token output reservation, less the 10% headroom. The AWS default
    # (30,000) is over it and was clamped, with a WARNING, on every query.
    "search": {"token_manager": {"max_context_tokens": LOCAL_MAX_CONTEXT_TOKENS}},
}


def _merge_under(preset: Any, data: Any) -> Any:
    """``data`` with ``preset`` filled in for every key ``data`` does not set."""
    if not isinstance(preset, dict) or not isinstance(data, dict):
        return copy.deepcopy(preset) if data is None else data
    merged = dict(data)
    for key, value in preset.items():
        merged[key] = _merge_under(value, data.get(key))
    return merged


def _strip_preset(preset: Any, data: Any) -> Any:
    """``data`` without the leaves equal to ``preset``'s (``_MISSING`` = none)."""
    if isinstance(preset, dict) and isinstance(data, dict):
        stripped = {}
        for key, value in data.items():
            kept = _strip_preset(preset.get(key, _MISSING), value)
            if kept is not _MISSING:
                stripped[key] = kept
        return stripped or _MISSING
    return _MISSING if preset is not _MISSING and data == preset else data


_MISSING: Any = object()


def strip_backend_preset(data: dict[str, Any], backend: Backend) -> dict[str, Any]:
    """Raw config ``data`` without the values the ``backend`` preset fills in.

    Undoes :func:`apply_backend_preset` on a loaded config, so switching it to
    another backend does not keep the old preset's values. A value equal to
    the preset's counts as the preset's, even when the file set it.
    """
    if backend is not Backend.LOCAL:
        return data
    stripped = _strip_preset(LOCAL_PRESET, data)
    return {} if stripped is _MISSING else dict(stripped)


def _kind_value(kind: Any) -> Any:
    return kind.value if isinstance(kind, Enum) else kind


def apply_backend_preset(data: dict[str, Any]) -> dict[str, Any]:
    """Raw config ``data`` with its ``backend`` preset filled in.

    Only ``backend: local`` has a preset; ``aws`` (the default) is the model
    defaults themselves. Values the config sets win, key by key. A config that
    authenticates to OpenSearch (``username`` or ``use_iam``) does not get the
    preset's anonymous access, which cannot be combined with either.
    """
    backend = data.get("backend", Backend.AWS)
    if isinstance(backend, Enum):
        backend = backend.value
    if backend != Backend.LOCAL.value:
        return data
    preset = copy.deepcopy(LOCAL_PRESET)
    opensearch = (data.get("aws") or {}).get("opensearch") or {}
    if isinstance(opensearch, dict) and (
        opensearch.get("username") or opensearch.get("use_iam")
    ):
        preset["aws"]["opensearch"].pop("allow_anonymous")
    # The context budget is derived from the default non-Bedrock window, so it
    # only fits a non-Bedrock LLM left at that window. Bedrock models (sized
    # from the catalog) or an explicit window keep the regular default.
    llm = (data.get("providers") or {}).get("llm") or {}
    if isinstance(llm, dict) and (
        _kind_value(llm.get("kind", preset["providers"]["llm"]["kind"])) == BEDROCK
        or "context_window_size" in llm
    ):
        preset.pop("search")
    merged = _merge_under(preset, data)
    assert isinstance(merged, dict)
    return merged
