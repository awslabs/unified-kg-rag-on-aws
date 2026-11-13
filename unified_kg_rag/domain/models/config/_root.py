# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The root ``Config`` and ``PipelineConfig``."""

import copy
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    model_validator,
)

from ..methodology import (
    METHODOLOGY_PROFILES,
    IndexArtifact,
    Methodology,
    artifact_switches,
    artifacts_of,
    buildable,
    switched_off_artifacts,
)
from ..provider_config import (
    BEDROCK,
    CROSS_ENCODER,
    GREMLIN,
    Backend,
    EmbeddingProviderConfig,
    LLMProviderConfig,
    ProvidersConfig,
    apply_backend_preset,
)
from ..retrieval import (
    validate_path_segment,
)
from ._aws import AWSConfig, RetryConfig
from ._common import ChunkLengthUnit, PipelineStageType, _AssignmentChecked
from ._evaluation import EvaluationConfig
from ._graph import GraphConfig
from ._indexing import IndexingConfig
from ._mcp import McpServerConfig
from ._merge import _changes, _plain, merge_over
from ._migration import _defined_keys_only, moved_keys, unknown_keys
from ._models import (
    DEFAULT_MODEL_ID,
    FAST_MODEL_ID,
    JUDGE_MODEL_ID,
    ROLE_TIERS,
    EmbeddingModelId,
    LanguageModelId,
    ModelRole,
    ModelsConfig,
    ModelTier,
    RerankModelId,
    _bare_model_id,
)
from ._processing import _CHUNK_SIZE_FIELDS, ChunkingConfig, ProcessingConfig
from ._prompts import CustomPromptConfig
from ._runtime import CacheConfig, LoggingConfig, MemoryConfig, ObservabilityConfig
from ._search import SearchConfig


class Config(_AssignmentChecked):
    """The whole configuration. Its ``backend`` preset is filled in when a
    config is validated from its input (the loader, ``Config(...)``,
    :meth:`updated`): assigning ``config.backend`` applies no preset, so
    build the config for another backend with
    ``config.updated({"backend": "local"})``."""

    backend: Backend = Field(
        default=Backend.AWS,
        description=(
            "Provider preset. 'aws' (default): Bedrock, Neptune, OpenSearch and "
            "DynamoDB. 'local': fills in local providers (Ollama, a cross-encoder "
            "reranker, embedded SQLite graph/vector stores and registry) for every "
            "providers/aws.neptune/aws.opensearch key the config leaves unset."
        ),
    )
    providers: ProvidersConfig = Field(
        default_factory=ProvidersConfig,
        description="Per-role provider selection (models, stores, registry)",
    )
    models: ModelsConfig = Field(
        default_factory=ModelsConfig,
        description="Which model serves each role, on the providers above",
    )
    aws: AWSConfig = Field(
        default_factory=AWSConfig, description="AWS services configuration"
    )
    processing: ProcessingConfig = Field(
        default_factory=ProcessingConfig, description="Text processing configuration"
    )
    graph: GraphConfig = Field(
        default_factory=GraphConfig, description="Graph analysis configuration"
    )
    indexing: IndexingConfig = Field(
        default_factory=IndexingConfig, description="Data indexing configuration"
    )
    search: SearchConfig = Field(
        default_factory=SearchConfig, description="Search configuration"
    )
    memory: MemoryConfig = Field(
        default_factory=MemoryConfig, description="Memory system configuration"
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig, description="Caching system configuration"
    )
    retry: RetryConfig = Field(
        default_factory=RetryConfig,
        description="Retries on transient errors, for every caller",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging system configuration"
    )
    observability: ObservabilityConfig = Field(
        default_factory=ObservabilityConfig,
        description="OpenTelemetry tracing configuration",
    )
    evaluation: EvaluationConfig = Field(
        default_factory=EvaluationConfig, description="Evaluation configuration"
    )
    custom_prompts: CustomPromptConfig = Field(
        default_factory=CustomPromptConfig, description="Custom prompt configuration"
    )
    mcp: McpServerConfig = Field(
        default_factory=McpServerConfig,
        description="The read-only MCP server of run-mcp-server",
    )

    # The input this config was validated from, before its backend preset
    # (see as_input).
    _input: dict[str, Any] | None = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _apply_backend_preset(
        cls, data: Any, handler: ModelWrapValidatorHandler["Config"]
    ) -> "Config":
        """Refuse the moved and unknown keys, fill in the ``backend`` preset
        under the keys the config sets, and keep the input it was given."""
        if not isinstance(data, dict):
            return handler(data)
        if moved := moved_keys(data):
            raise ValueError("settings moved; replace " + "; ".join(moved))
        if unknown := unknown_keys(data, cls):
            # A misspelled key would leave its setting at the default (a
            # guardrail identifier under the wrong key leaves queries
            # unscreened). ConfigLoader(allow_unknown_keys=True) strips them.
            raise ValueError(
                "unknown config keys: " + ", ".join(unknown) + " (fix or remove them)"
            )
        # What is left that no section defines is a removed setting, ignored
        # (the loader warns of it): the sections refuse a key they do not
        # define.
        config = handler(apply_backend_preset(_defined_keys_only(data, cls)))
        config._input = _plain(data)
        return config

    def __eq__(self, other: object) -> bool:
        """Equal settings make equal configs, whatever input they came from."""
        if not isinstance(other, Config):
            return NotImplemented
        return type(self) is type(other) and self.__dict__ == other.__dict__

    def as_input(self) -> dict[str, Any]:
        """Input that validates to this config again, without its preset.

        The input it was validated from (the file and the environment, as
        given; not the ``backend`` preset's values, which validation fills
        in again), with every change made to the object since written over
        it. Rebuilding from it under another ``backend`` or provider keeps
        what the caller set, even a value equal to a preset's, and drops
        only what the old preset filled in.
        """
        if self._input is None:
            return self.model_dump(exclude_unset=True)
        built = type(self).model_validate(copy.deepcopy(self._input))
        return merge_over(copy.deepcopy(self._input), _changes(built, self))

    def updated(
        self,
        changes: Mapping[str, Any] | None = None,
        *,
        overlay: Mapping[str, Any] | None = None,
    ) -> "Config":
        """A new config: this one with ``overlay`` merged over it (nested
        keys as in a config file: ``{"processing": {"graph_extraction":
        {...}}}``, a mapping merged into its section) and then each dotted
        key of ``changes`` set (``{"indexing.methodologies": ["graphrag"]}``,
        the value replacing the setting's), validated whole, so the checks
        across sections run as they do at load (an assignment validates only
        its own section). The one way to build a changed config. This config
        is unchanged.

        Raises:
            ValueError: A key of ``changes`` names no setting.
            pydantic.ValidationError: The changed config is invalid (an
                unknown key of ``overlay`` included).
        """
        data = self.as_input()
        if overlay:
            data = merge_over(data, _plain(dict(overlay)))
        for dotted, value in (changes or {}).items():
            model: type[BaseModel] | None = type(self)
            node = data
            parts = dotted.split(".")
            for depth, part in enumerate(parts):
                field = model.model_fields.get(part) if model is not None else None
                if model is not None and field is None:
                    raise ValueError(f"No setting '{dotted}'")
                if depth == len(parts) - 1:
                    node[part] = _plain(value)
                    break
                annotation = field.annotation if field is not None else None
                model = (
                    annotation
                    if isinstance(annotation, type)
                    and issubclass(annotation, BaseModel)
                    else None
                )
                child = node.get(part)
                if not isinstance(child, dict):
                    child = _plain(child) if isinstance(child, BaseModel) else {}
                    node[part] = child
                node = child
        return type(self).model_validate(data)

    @model_validator(mode="after")
    def _check_models(self) -> "Config":
        """Check the models against the provider of their family.

        A Bedrock provider takes only catalog members for embedding and
        rerank (the vector dimension and rerank limits come from the
        catalog); another provider takes any name but needs one.
        """
        providers, models = self.providers, self.models
        errors: list[str] = []
        if providers.llm.kind != BEDROCK and models.default is None:
            errors.append(
                "models.default is required when providers.llm.kind is "
                f"'{providers.llm.kind}'"
            )
        embedding = models.embedding
        if providers.embedding.kind == BEDROCK:
            if embedding.model_id is not None and not isinstance(
                embedding.model_id, EmbeddingModelId
            ):
                errors.append(
                    f"models.embedding.model_id: '{embedding.model_id}' is not a "
                    "Bedrock embedding model; expected one of "
                    f"{[m.value for m in EmbeddingModelId]} (or select another "
                    "providers.embedding.kind)"
                )
        elif embedding.model_id is None or embedding.dimension is None:
            errors.append(
                "models.embedding.model_id and models.embedding.dimension are "
                "required when providers.embedding.kind is "
                f"'{providers.embedding.kind}'"
            )
        if providers.rerank.kind == BEDROCK:
            if models.rerank is not None and not isinstance(
                models.rerank, RerankModelId
            ):
                errors.append(
                    f"models.rerank: '{models.rerank}' is not a Bedrock rerank "
                    f"model; expected one of {[m.value for m in RerankModelId]} "
                    "(or select another providers.rerank.kind)"
                )
        elif providers.rerank.kind == CROSS_ENCODER and models.rerank is None:
            errors.append(
                "models.rerank is required when providers.rerank.kind is "
                f"'{CROSS_ENCODER}'"
            )
        if providers.graph.kind == GREMLIN and self.aws.neptune.use_iam:
            errors.append(
                "providers.graph.kind 'gremlin' needs aws.neptune.use_iam: false "
                "(SigV4 signing is Neptune-only)"
            )
        if errors:
            raise ValueError("; ".join(errors))
        return self

    @model_validator(mode="after")
    def _check_methodologies(self) -> "Config":
        """Refuse a selection a switch leaves nothing of (every selected
        methodology needs an artifact a switch turns off)."""
        if not self.effective_methodologies:
            off = {
                a
                for m in self.indexing.methodologies
                for a in METHODOLOGY_PROFILES[m].required & self.artifacts_switched_off
            }
            raise ValueError(
                "indexing.methodologies ("
                + ", ".join(m.value for m in self.indexing.methodologies)
                + f") needs {', '.join(sorted(a.value for a in off))}, which "
                + " and ".join(self._artifact_switches(off))
                + " turns off: select another methodology or turn it back on"
            )
        return self

    def _switch_on(self, key: str) -> bool:
        """The value of the boolean setting at dotted ``key``
        (``ARTIFACT_SWITCHES``)."""
        value: Any = self
        for part in key.split("."):
            value = getattr(value, part)
        return bool(value)

    @property
    def artifacts_switched_off(self) -> frozenset[IndexArtifact]:
        """The index artifacts a switch set false turns off: no run of this
        config builds them (``ARTIFACT_SWITCHES``)."""
        return switched_off_artifacts(self._switch_on)

    @staticmethod
    def _artifact_switches(artifacts: Collection[IndexArtifact]) -> list[str]:
        """The switches that turn ``artifacts`` off, as a message names them."""
        return [f"{key}: false" for key in artifact_switches(artifacts)]

    @property
    def effective_methodologies(self) -> tuple[Methodology, ...]:
        """``indexing.methodologies`` without those whose required artifacts a
        switch turns off (docs/design.md#46-methodology-profiles): what
        ingestion builds (without a build record, a query takes the namespace
        to hold every methodology whose required artifacts these build)."""
        return buildable(self.indexing.methodologies, self.artifacts_switched_off)

    # --- resolved models -----------------------------------------------------

    def tier_model_id(self, tier: ModelTier) -> str:
        """The model of ``tier``: ``models.<tier>``, else the provider's default."""
        models = self.models
        default = models.default or DEFAULT_MODEL_ID
        if tier == "default":
            return default
        if models.fast is not None:
            return models.fast
        return FAST_MODEL_ID if self.providers.llm.kind == BEDROCK else default

    def model_id(self, role: ModelRole) -> str:
        """The model that serves ``role``: its ``models.roles`` entry (a model
        id or a tier name), else its tier's model (the evaluation judge on
        Bedrock: ``JUDGE_MODEL_ID``)."""
        chosen: str | None = getattr(self.models.roles, role.value)
        if chosen is None:
            if role is ModelRole.EVALUATION:
                return self.judge_model_choice()[0]
            return self.tier_model_id(ROLE_TIERS[role])
        if chosen == "default":
            return self.tier_model_id("default")
        if chosen == "fast":
            return self.tier_model_id("fast")
        return chosen

    def judge_model_choice(
        self, answer_models: Collection[str] | None = None
    ) -> tuple[str, str]:
        """The run-eval judge (``models.roles.evaluation``) and why it is the
        one.

        A configured judge is used as given, even when it is the answer
        model (run-eval then warns). Unset on Bedrock, it is
        ``JUDGE_MODEL_ID`` unless that is the answer model, then the first
        of the default tier's model and Claude Sonnet 5.5 that is not: a
        judge favours its own answers. ``answer_models`` are the models the
        judged answers come from (saved runs a re-judge reads); ``None``:
        the configured answer model. With another provider it is the default tier's model, the only one
        the config names.
        """
        configured: str | None = self.models.roles.evaluation
        if configured is not None:
            return self.model_id_for_choice(configured), "configured"
        if self.providers.llm.kind != BEDROCK:
            return self.tier_model_id("default"), "the default tier (not Bedrock)"
        if answer_models is None:
            answer_models = [self.model_id(ModelRole.ANSWER_GENERATION)]
        answers = {_bare_model_id(model) for model in answer_models}
        if _bare_model_id(JUDGE_MODEL_ID) not in answers:
            return JUDGE_MODEL_ID, "the default judge, not the answer model"
        for candidate in (
            self.tier_model_id("default"),
            LanguageModelId.CLAUDE_V5_5_SONNET.value,
        ):
            if _bare_model_id(candidate) not in answers:
                return candidate, (
                    f"the default judge {JUDGE_MODEL_ID} is the answer model"
                )
        return JUDGE_MODEL_ID, "no other model to judge with"

    def model_id_for_choice(self, chosen: str) -> str:
        """A ``models.roles`` value resolved: a tier name to its model."""
        if chosen == "default":
            return self.tier_model_id("default")
        if chosen == "fast":
            return self.tier_model_id("fast")
        return chosen

    def model_tier(self, model_id: str) -> ModelTier:
        """Tier a call on ``model_id`` belongs to, for picking its effort.

        Only a model that is the fast tier's model (and not also the default
        tier's) counts as fast; anything else, including a role pinned to its
        own model, gets the default tier's effort.
        """
        model_id = str(model_id).strip()
        fast = self.tier_model_id("fast")
        if model_id == fast and fast != self.tier_model_id("default"):
            return "fast"
        return "default"

    @property
    def embedding_model_id(self) -> EmbeddingModelId | str:
        """The embedding model (``models.embedding.model_id``; Bedrock default
        Titan Embed V2)."""
        model_id = self.models.embedding.model_id
        return EmbeddingModelId.TITAN_EMBED_V2 if model_id is None else model_id

    @property
    def embedding_dimension(self) -> int | None:
        """``models.embedding.dimension`` (``None``: the Bedrock model's largest)."""
        return self.models.embedding.dimension

    @property
    def rerank_model_id(self) -> RerankModelId | str | None:
        """The rerank model (``models.rerank``; Bedrock default Cohere Rerank
        3.5); ``None`` with a reranker that takes no model."""
        if self.models.rerank is not None:
            return self.models.rerank
        if self.providers.rerank.kind == BEDROCK:
            return RerankModelId.COHERE_RERANK_V3_5
        return None

    def inactive_provider_settings(self) -> list[str]:
        """Settings the config sets that its providers do not read.

        The endpoint and sampling settings of ``providers.llm`` while the
        language models are Bedrock's, the Bedrock model knobs while they are
        not, the endpoint settings of ``providers.embedding`` while
        embeddings are Bedrock's, and the cross-encoder settings of
        ``providers.rerank`` under another reranker. Logged as a WARNING at
        load: such a setting is ignored.
        """
        providers = self.providers
        sections: list[tuple[str, BaseModel, frozenset[str], bool]] = [
            (
                "providers.llm",
                providers.llm,
                frozenset(LLMProviderConfig.model_fields) - {"kind"},
                providers.llm.kind == BEDROCK,
            ),
            (
                "aws.bedrock",
                self.aws.bedrock,
                frozenset(
                    {
                        "default_effort",
                        "fast_effort",
                        "ingestion_effort",
                        "enable_1m_context",
                        "model_overrides",
                    }
                ),
                providers.llm.kind != BEDROCK,
            ),
            (
                "providers.embedding",
                providers.embedding,
                frozenset(EmbeddingProviderConfig.model_fields) - {"kind"},
                providers.embedding.kind == BEDROCK,
            ),
            (
                "providers.rerank",
                providers.rerank,
                frozenset({"device", "max_length", "batch_size"}),
                providers.rerank.kind != CROSS_ENCODER,
            ),
        ]
        return [
            f"{prefix}.{name}"
            for prefix, section, names, inactive in sections
            if inactive
            for name in sorted(names & section.model_fields_set)
        ]

    def ineffective_settings(self) -> list[str]:
        """Why settings the config sets do not do what they say, one message
        each. Logged as a WARNING at load: the config is valid, but likely
        not what was meant.

        - Contextual grounding on without a guardrail identifier: no
          guardrail screens the answers, so no grounding check runs.
        - ``length_unit: chars`` with every chunk size at its token
          default: the sizes are read as characters, about a quarter of the
          intended chunk length in English text.
        - Community reports off with graphrag selected: global and drift,
          which answer from them, are refused on what ingestion builds.
        """
        messages: list[str] = []
        guardrail = self.aws.bedrock.guardrail
        if guardrail.contextual_grounding.enabled and not guardrail.enabled:
            messages.append(
                "aws.bedrock.guardrail.contextual_grounding.enabled is set "
                "without aws.bedrock.guardrail.identifier: no guardrail screens "
                "the answers, so no grounding check runs"
            )
        chunking = self.processing.chunking
        if chunking.length_unit is ChunkLengthUnit.CHARS and all(
            getattr(chunking, name) == ChunkingConfig.model_fields[name].default
            for name in _CHUNK_SIZE_FIELDS
        ):
            messages.append(
                "processing.chunking.length_unit is chars, but every chunk size "
                "is still its token default (max_chunk_size 2000 characters is "
                "about 500 English tokens): set the sizes in characters, "
                "for example min 1000, max 8000, overlap 500, fallback 4800, "
                "pre-chunk 50000 and its overlap 500"
            )
        effective = self.effective_methodologies
        # A config that sets no methodologies but switches an artifact off
        # (a LightRAG-only config from before they were selectable) builds
        # the profile it leaves: nothing to warn of.
        dropped = [m for m in self.indexing.methodologies if m not in effective]
        if dropped and "methodologies" in self.indexing.model_fields_set:
            off = {
                a
                for m in dropped
                for a in METHODOLOGY_PROFILES[m].required & self.artifacts_switched_off
            }
            messages.append(
                "indexing.methodologies selects "
                + ", ".join(m.value for m in dropped)
                + ", but "
                + " and ".join(self._artifact_switches(off))
                + " turns off what it needs: ingestion builds "
                + ", ".join(m.value for m in effective)
                + " only, and queries of its strategies are refused"
            )
        if self.processing.claim_extraction.enabled and IndexArtifact.CLAIMS not in (
            artifacts_of(effective)
        ):
            messages.append(
                "processing.claim_extraction is enabled, but no selected "
                "methodology reads claims (graphrag's local does): ingestion "
                "still extracts them"
            )
        if (
            Methodology.GRAPHRAG in effective
            and IndexArtifact.COMMUNITY_REPORTS in self.artifacts_switched_off
            and self.graph.community_detection.enabled
        ):
            messages.append(
                "graph.community_detection.report_generation.enabled is false: "
                "ingestion builds no community reports, so global and drift "
                "search, which answer from them, are refused (local reads "
                "graphrag without them)"
            )
        return messages

    @property
    def doc_status_enabled(self) -> bool:
        """Whether incremental indexing (the doc-status registry) is on:
        ``providers.doc_status.enabled``."""
        return self.providers.doc_status.enabled


class PipelineConfig(BaseModel):
    """How one ingestion run executes (stages, caching, resume); an unknown
    key is refused, not ignored."""

    model_config = ConfigDict(extra="forbid")

    stages_enabled: dict[PipelineStageType, bool] = Field(
        default_factory=lambda: dict.fromkeys(PipelineStageType, True),
        description="Specifies which pipeline stages are enabled.",
    )
    cache_enabled: bool = Field(
        default=True, description="Enables caching for pipeline stage outputs."
    )
    local_directory: str | Path = Field(
        default="cache",
        description="The local filesystem path for cache storage.",
    )
    s3_sync_enabled: bool = Field(
        default=False,
        description="Enables synchronization of the cache with an S3 bucket.",
    )
    s3_bucket_name: str | None = Field(
        default=None, description="The S3 bucket name for cache synchronization."
    )
    s3_prefix: str = Field(
        default="pipeline-runs",
        min_length=1,
        description="The S3 key prefix for storing cache objects.",
    )
    continue_on_error: bool = Field(
        default=False,
        description="If true, the pipeline continues execution even if a stage fails.",
    )
    force_rebuild: bool = Field(
        default=False,
        description=(
            "If true, ignores the cached stage outputs, the stored extraction "
            "records and the LLM response cache and rebuilds all outputs: every "
            "model call and embedding is paid again."
        ),
    )
    # The id names a local cache directory and an S3 prefix.
    pipeline_id: (
        Annotated[
            str, AfterValidator(lambda v: validate_path_segment(v, "pipeline_id"))
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "The ID of a previous pipeline run to resume (lowercase letters, "
            "digits, hyphens and underscores)."
        ),
    )
    resume_from_stage: str | None = Field(
        default=None,
        min_length=1,
        description="The name of the stage from which to resume execution.",
    )
