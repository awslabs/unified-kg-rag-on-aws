# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model identifiers and the ``models`` section (roles, tiers, embeddings)."""

import re
from collections.abc import Callable
from enum import Enum
from typing import Annotated, Any, Literal, get_args

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    StringConstraints,
    field_validator,
)


class EmbeddingModelId(str, Enum):
    """The embedding models the package supports; a closed list.

    Unlike language-model ids, embedding-model config fields accept only these
    members: the model's vector dimension is fixed into the OpenSearch
    ``knn_vector`` mappings and validated against its capability record, so an
    unknown id has no safe default. Adding a model takes a member here plus a
    record in ``adapters/aws/bedrock_models.py``.
    """

    EMBED_MULTILINGUAL_V3 = "cohere.embed-multilingual-v3"
    EMBED_V4 = "cohere.embed-v4:0"
    EMBED_ENGLISH_V3 = "cohere.embed-english-v3"
    TITAN_EMBED_V1 = "amazon.titan-embed-text-v1"
    TITAN_EMBED_V2 = "amazon.titan-embed-text-v2:0"
    # NOTE: add new models here


class LanguageModelId(str, Enum):
    """Named constants for the language models with a curated capability record.

    Model-id config fields accept ANY Bedrock model id string, so a new model
    can be tried without a release; these members are only convenience names
    (and the keys of the adapter's capability table). ``str()`` and
    f-strings render the bare id, so a member can be used wherever an id
    string is expected.
    """

    # Claude 4.7+ / 5.x ids carry no date suffix and no ':0' revision, unlike
    # earlier generations. Cross-region resolution still applies the same
    # 'us.'/'apac.'/'global.' inference-profile prefixes.
    CLAUDE_V5_5_SONNET = "anthropic.claude-sonnet-5-5"
    CLAUDE_V5_5_OPUS = "anthropic.claude-opus-5-5"
    CLAUDE_V5_5_HAIKU = "anthropic.claude-haiku-5-5"
    CLAUDE_V5_SONNET = "anthropic.claude-sonnet-5"
    CLAUDE_V5_OPUS = "anthropic.claude-opus-5"
    CLAUDE_V4_8_OPUS = "anthropic.claude-opus-4-8"
    CLAUDE_V4_7_OPUS = "anthropic.claude-opus-4-7"
    CLAUDE_V4_6_OPUS = "anthropic.claude-opus-4-6-v1"
    CLAUDE_V4_6_SONNET = "anthropic.claude-sonnet-4-6"
    CLAUDE_V3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_V3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_V3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    CLAUDE_V3_5_HAIKU = "anthropic.claude-3-5-haiku-20241022-v1:0"
    CLAUDE_V4_5_HAIKU = "anthropic.claude-haiku-4-5-20251001-v1:0"
    CLAUDE_V3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_V3_5_SONNET_V2 = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_V3_7_SONNET = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    CLAUDE_V4_SONNET = "anthropic.claude-sonnet-4-20250514-v1:0"
    CLAUDE_V4_5_SONNET = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    CLAUDE_V4_OPUS = "anthropic.claude-opus-4-20250514-v1:0"
    CLAUDE_V4_1_OPUS = "anthropic.claude-opus-4-1-20250805-v1:0"
    CLAUDE_V4_5_OPUS = "anthropic.claude-opus-4-5-20251101-v1:0"
    # OpenAI proprietary GPT models (open-weight gpt-oss is intentionally not
    # offered). Ids use dotted versions and are served through the Converse API
    # on cross-region inference profiles only.
    GPT_V6_1_SOL = "openai.gpt-6.1-sol"
    GPT_V6_ASTRA = "openai.gpt-6-astra"
    GPT_V6_SOL = "openai.gpt-6-sol"
    GPT_V6_LUNA = "openai.gpt-6-luna"
    GPT_V5_6_SOL = "openai.gpt-5.6-sol"
    GPT_V5_6_TERRA = "openai.gpt-5.6-terra"
    GPT_V5_6_LUNA = "openai.gpt-5.6-luna"
    GPT_V5_5 = "openai.gpt-5.5"
    GPT_V5_4 = "openai.gpt-5.4"
    # NOTE: add new models here

    def __str__(self) -> str:
        return str(self.value)


ModelTier = Literal["default", "fast"]
EffortLevel = Literal["low", "medium", "high", "xhigh", "max"]
EFFORT_LEVELS: frozenset[str] = frozenset(get_args(EffortLevel))
# Reasoning tokens an adaptive-thinking (or OpenAI reasoning) request is
# budgeted at each effort level when an output floor or ceiling is sized:
# thinking counts toward max_tokens, so a limit below it would cut the
# answer off.
EFFORT_THINKING_TOKENS: dict[str, int] = {
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
    "max": 32768,
}

# The Bedrock models of the two tiers when models.default / models.fast are
# unset, so changing a tier default is a one-line edit here.
DEFAULT_MODEL_ID: str = LanguageModelId.CLAUDE_V5_5_SONNET.value
FAST_MODEL_ID: str = LanguageModelId.CLAUDE_V5_5_HAIKU.value
# The run-eval judge (models.roles.evaluation) when unset, on Bedrock: a model
# other than the default answer model, since an LLM judge favours its own
# answers (measured: Sonnet 5.5 judging its own answers against Opus 5.5's
# moved the directness win rate by -0.27 against a cross-family judge).
JUDGE_MODEL_ID: str = LanguageModelId.CLAUDE_V5_5_OPUS.value


# The cross-region inference-profile prefixes a configured model id may carry.
_PROFILE_PREFIX = re.compile(r"^(?:global|us|us-gov|eu|apac|jp|au|ca)\.")


def _bare_model_id(model_id: str) -> str:
    """``model_id`` without an inference-profile prefix, for comparing two
    configured models."""
    return _PROFILE_PREFIX.sub("", str(model_id).strip())


def _coerce_model_id(value: Any) -> Any:
    # A LanguageModelId member (or any str enum) validates as its bare id.
    if isinstance(value, Enum):
        value = value.value
    return value.strip() if isinstance(value, str) else value


class ModelPurpose(str, Enum):
    """Why a language model is being created, so per-path policies can differ.

    Callers pass it to the LLM factory (``get_model(model_purpose=...)``):
    the response cache and the ingestion effort apply to ``INGESTION``
    models, and query chains get a transient-error retry. Unspecified means
    ``QUERY``.
    """

    QUERY = "query"
    INGESTION = "ingestion"
    EVALUATION = "evaluation"


class RerankModelId(str, Enum):
    """The rerank models the package supports; a closed list.

    Like ``EmbeddingModelId``, and unlike language-model ids, only these
    members are accepted. Adding a model takes a member here plus a record
    (document and token limits) in ``adapters/aws/bedrock_models.py``.
    """

    AMAZON_RERANK_V1 = "amazon.rerank-v1:0"
    COHERE_RERANK_V3_5 = "cohere.rerank-v3-5:0"
    # NOTE: add new models here


def model_id_value(model_id: Any) -> str:
    """The bare id string of a model-id config value (enum member or str)."""
    return str(model_id.value) if isinstance(model_id, Enum) else str(model_id)


def _known_member(enum_cls: type[Enum]) -> Callable[[Any], Any]:
    """Coerce a string naming a member of ``enum_cls`` to that member."""

    def coerce(value: Any) -> Any:
        if isinstance(value, enum_cls):
            return value
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, str):
            value = value.strip()
            try:
                return enum_cls(value)
            except ValueError:
                return value
        return value

    return coerce


# Embedding/rerank model ids: a catalog member when the id is one, else the
# bare string. A non-Bedrock provider serves any model name; for Bedrock the
# Config validator still requires a catalog member (the closed lists above).
EmbeddingModelRef = Annotated[
    EmbeddingModelId | Annotated[str, StringConstraints(min_length=1)],
    Field(union_mode="left_to_right"),
    BeforeValidator(_known_member(EmbeddingModelId)),
]
RerankModelRef = Annotated[
    RerankModelId | Annotated[str, StringConstraints(min_length=1)],
    Field(union_mode="left_to_right"),
    BeforeValidator(_known_member(RerankModelId)),
]


class ModelRole(str, Enum):
    """Every place the framework calls a language model, by its role.

    Each role runs on a tier (``ROLE_TIERS``) unless ``models.roles`` names
    another model or tier for it.
    """

    FIXING = "fixing"
    CHUNKING = "chunking"
    TRANSLATION = "translation"
    GRAPH_EXTRACTION = "graph_extraction"
    DESCRIPTION_SUMMARY = "description_summary"
    GLEANING = "gleaning"
    CLAIM_EXTRACTION = "claim_extraction"
    COMMUNITY_REPORT = "community_report"
    SUMMARY_TREE = "summary_tree"
    QUERY_TRANSLATION = "query_translation"
    ENTITY_EXTRACTION = "entity_extraction"
    STRATEGY_SELECTION = "strategy_selection"
    QUERY_REWRITE = "query_rewrite"
    ANSWER_GENERATION = "answer_generation"
    COMMUNITY_RELEVANCE = "community_relevance"
    GLOBAL_MAP = "global_map"
    DRIFT_PRIMER = "drift_primer"
    DRIFT_ACTION = "drift_action"
    QUERY_CONTROLLER = "query_controller"
    GRAPH_QUERY = "graph_query"
    EVALUATION = "evaluation"
    ANSWER_SPAN = "answer_span"
    PAIRWISE_OVERLAP = "pairwise_overlap"


_ROLE_DESCRIPTIONS: dict[ModelRole, tuple[ModelTier, str]] = {
    ModelRole.FIXING: ("default", "repairs malformed structured model output"),
    ModelRole.CHUNKING: ("fast", "places chunk boundaries (intelligent chunker)"),
    ModelRole.TRANSLATION: ("fast", "translates the corpus at ingestion"),
    ModelRole.GRAPH_EXTRACTION: ("default", "extracts entities and relationships"),
    ModelRole.DESCRIPTION_SUMMARY: (
        "fast",
        "summarizes the merged descriptions of an entity or relationship",
    ),
    ModelRole.GLEANING: ("default", "refines each text unit's extraction"),
    ModelRole.CLAIM_EXTRACTION: ("default", "extracts claims"),
    ModelRole.COMMUNITY_REPORT: ("default", "writes community reports"),
    ModelRole.SUMMARY_TREE: (
        "default",
        "summarizes each cluster of the summary tree (raptor methodology)",
    ),
    ModelRole.QUERY_TRANSLATION: ("fast", "translates the query"),
    ModelRole.ENTITY_EXTRACTION: (
        "default",
        "extracts the query's entities and keywords",
    ),
    ModelRole.STRATEGY_SELECTION: ("fast", "routes an auto query to a strategy"),
    ModelRole.QUERY_REWRITE: ("fast", "rewrites a follow-up into a standalone query"),
    ModelRole.ANSWER_GENERATION: ("default", "writes the answer"),
    ModelRole.COMMUNITY_RELEVANCE: (
        "fast",
        "rates community relevance for global search's dynamic selection",
    ),
    ModelRole.GLOBAL_MAP: ("fast", "answers from each report batch (global map)"),
    ModelRole.DRIFT_PRIMER: ("fast", "writes DRIFT's primer and follow-ups"),
    ModelRole.DRIFT_ACTION: ("fast", "answers each DRIFT follow-up"),
    ModelRole.QUERY_CONTROLLER: (
        "fast",
        "judges whether iterative search's context answers the question and "
        "names the follow-up queries",
    ),
    ModelRole.GRAPH_QUERY: (
        "default",
        "writes the structured strategy's graph query plan (search.structured)",
    ),
    ModelRole.EVALUATION: ("default", "judges answers in run-eval"),
    ModelRole.ANSWER_SPAN: (
        "fast",
        "extracts the short answer span run-eval scores exact match and token "
        "F1 on (evaluation.answer_span: extracted)",
    ),
    ModelRole.PAIRWISE_OVERLAP: (
        "fast",
        "splits two answers into shared and unique statements for run-eval "
        "--pairwise-mode differential",
    ),
}
# The tier each role runs on unless models.roles says otherwise.
ROLE_TIERS: dict[ModelRole, ModelTier] = {
    role: tier for role, (tier, _) in _ROLE_DESCRIPTIONS.items()
}


def _role_field(role: ModelRole) -> Any:
    tier, what = _ROLE_DESCRIPTIONS[role]
    return Field(
        default=None,
        min_length=1,
        description=(
            f"Model that {what}: a model id, or a tier name ('default' / "
            f"'fast'). null runs it on the '{tier}' tier."
        ),
    )


_EVALUATION_ROLE_FIELD: Any = Field(
    default=None,
    min_length=1,
    description=(
        "Model of the run-eval LLM judges (the ragas and assertion evaluators "
        "and --pairwise-judge): a model id, or a tier name ('default' / "
        f"'fast'). null = {JUDGE_MODEL_ID} on Bedrock, a model other than "
        "the default answer model; when the answer model is that model, the "
        "default tier's model or else Claude Sonnet 5.5, whichever is not "
        "(run-eval logs the choice at INFO): a judge favours its own answers (measured "
        "on 50 global questions: Sonnet 5.5 judging its own answers against "
        "Opus 5.5's shifted the directness win rate by -0.27 [-0.38, -0.16] "
        "against a GPT-5.5 judge). It costs about twice the default tier per "
        "token (on-demand, us-west-2, 2026-11: $4 / $20 per million input / "
        "output tokens against $2 / $10). With another provider, null runs "
        "it on the 'default' tier. 'default' restores the judge of earlier "
        "versions, the answer model (run-eval warns when a configured judge "
        "is the answer model)."
    ),
)


class RoleModelsConfig(BaseModel):
    """Per-role model overrides (``models.roles``)."""

    fixing: str | None = _role_field(ModelRole.FIXING)
    chunking: str | None = _role_field(ModelRole.CHUNKING)
    translation: str | None = _role_field(ModelRole.TRANSLATION)
    graph_extraction: str | None = _role_field(ModelRole.GRAPH_EXTRACTION)
    description_summary: str | None = _role_field(ModelRole.DESCRIPTION_SUMMARY)
    gleaning: str | None = _role_field(ModelRole.GLEANING)
    claim_extraction: str | None = _role_field(ModelRole.CLAIM_EXTRACTION)
    community_report: str | None = _role_field(ModelRole.COMMUNITY_REPORT)
    summary_tree: str | None = _role_field(ModelRole.SUMMARY_TREE)
    query_translation: str | None = _role_field(ModelRole.QUERY_TRANSLATION)
    entity_extraction: str | None = _role_field(ModelRole.ENTITY_EXTRACTION)
    strategy_selection: str | None = _role_field(ModelRole.STRATEGY_SELECTION)
    query_rewrite: str | None = _role_field(ModelRole.QUERY_REWRITE)
    answer_generation: str | None = _role_field(ModelRole.ANSWER_GENERATION)
    community_relevance: str | None = _role_field(ModelRole.COMMUNITY_RELEVANCE)
    global_map: str | None = _role_field(ModelRole.GLOBAL_MAP)
    drift_primer: str | None = _role_field(ModelRole.DRIFT_PRIMER)
    drift_action: str | None = _role_field(ModelRole.DRIFT_ACTION)
    query_controller: str | None = _role_field(ModelRole.QUERY_CONTROLLER)
    graph_query: str | None = _role_field(ModelRole.GRAPH_QUERY)
    evaluation: str | None = _EVALUATION_ROLE_FIELD
    answer_span: str | None = _role_field(ModelRole.ANSWER_SPAN)
    pairwise_overlap: str | None = _role_field(ModelRole.PAIRWISE_OVERLAP)

    @field_validator("*", mode="before")
    @classmethod
    def _bare_id(cls, value: Any) -> Any:
        return _coerce_model_id(value)


class EmbeddingModelConfig(BaseModel):
    """The one embedding model (``models.embedding``): index, queries,
    evaluation and the visualization layout all embed with it."""

    model_id: EmbeddingModelRef | None = Field(
        default=None,
        description=(
            "Embedding model. With the bedrock embedding provider one of the "
            "EmbeddingModelId values (the vector dimension must be known for "
            "the index mapping), null = amazon.titan-embed-text-v2:0; with "
            "another provider any model name, required."
        ),
    )
    dimension: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Vector dimension, fixed into the index mappings. Bedrock: one the "
            "model supports (Titan Embed V2: 256, 512 or 1024), null = its "
            "largest. Another provider: the model's dimension, required."
        ),
    )


class ModelsConfig(BaseModel):
    """Which model serves each role (``models``).

    The provider of each family is ``providers.llm.kind`` /
    ``providers.embedding.kind`` / ``providers.rerank.kind``; the ids here are
    checked against it. Read the resolved models through ``Config.model_id``,
    ``Config.embedding_model_id`` and ``Config.rerank_model_id``.
    """

    default: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Model of the 'default' tier (extraction, gleaning, claims, "
            "community reports, output fixing, query entity/keyword "
            "extraction, answer generation; evaluation with another "
            "provider than Bedrock, see roles.evaluation). null = "
            f"{DEFAULT_MODEL_ID} on Bedrock; required with another provider."
        ),
    )
    fast: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Model of the 'fast' tier (chunking, translation, description "
            "summarization, global/DRIFT search steps, query translation, "
            f"rewriting and routing). null = {FAST_MODEL_ID} on Bedrock, the "
            "default tier's model with another provider."
        ),
    )
    roles: RoleModelsConfig = Field(
        default_factory=RoleModelsConfig,
        description="Per-role overrides: a model id or a tier name per role.",
    )
    embedding: EmbeddingModelConfig = Field(
        default_factory=EmbeddingModelConfig,
        description="The embedding model and its dimension.",
    )
    rerank: RerankModelRef | None = Field(
        default=None,
        description=(
            "Rerank model. Bedrock: one of the RerankModelId values, null = "
            "cohere.rerank-v3-5:0. cross_encoder: a sentence-transformers "
            "model name, required. Ignored with providers.rerank.kind none."
        ),
    )
    default_max_output_tokens: int | None = Field(
        default=16384,
        ge=1,
        description=(
            "max_tokens sent with each LLM request, clamped to the model's "
            "maximum (providers.llm.max_output_tokens for a non-Bedrock "
            "model). Bedrock reserves input + max_tokens against the "
            "tokens-per-minute quota when a request starts, so sending the "
            "model maximum (128K on Claude 5.x) throttles concurrent calls "
            "long before real usage does. Thinking tokens count toward it. "
            "Prompts with long outputs (graph/claim extraction, gleaning, "
            "community reports, document translation) declare a higher floor "
            "that wins over this value. Raise it if answers are cut off "
            "(stopReason max_tokens); null sends the model maximum."
        ),
    )

    @field_validator("default", "fast", mode="before")
    @classmethod
    def _bare_id(cls, value: Any) -> Any:
        return _coerce_model_id(value)
