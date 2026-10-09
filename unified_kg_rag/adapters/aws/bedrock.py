# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar, Generic, Literal, TypeVar
from uuid import UUID

import boto3
from aws_assume_role_lib.aws_assume_role_lib import assume_role
from botocore.config import Config as BotoConfig
from langchain_aws import BedrockEmbeddings, ChatBedrock, ChatBedrockConverse
from langchain_aws.document_compressors.rerank import BedrockRerank
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
from langchain_core.documents import Document
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import Field, PrivateAttr

from unified_kg_rag.adapters.aws.bedrock_models import (
    _EMBEDDING_MODEL_INFO,
    _RERANK_MODEL_INFO,
    EmbeddingModelInfo,
    LanguageModelInfo,
    RerankModelInfo,
    base_model_id,
    effective_max_output_tokens,
    get_language_model_info,
)
from unified_kg_rag.adapters.aws.bedrock_retry import call_with_transient_retry
from unified_kg_rag.adapters.aws.token_counter import BedrockTokenCounter
from unified_kg_rag.domain.models import (
    Config,
    EmbeddingModelId,
    ModelPurpose,
    RerankModelId,
)
from unified_kg_rag.domain.models.config import (
    EFFORT_LEVELS,
    ModelTier,
    TransientRetryConfig,
)
from unified_kg_rag.shared import (
    AWSServiceError,
    EmbeddingModelError,
    LanguageModelError,
    RerankModelError,
    get_logger,
)

logger = get_logger(__name__)


DEFAULT_ROLE_SESSION_NAME: str = "unified-kg-rag-on-aws-role-session"


ModelIdT = TypeVar("ModelIdT")
ModelInfoT = TypeVar("ModelInfoT")
WrapperT = TypeVar("WrapperT")


class BaseBedrockWrapper:
    _token_counter: BedrockTokenCounter | None = PrivateAttr(default=None)

    def __init__(
        self, token_counter: BedrockTokenCounter | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._token_counter = token_counter

    @property
    def _buffer_tokens(self) -> int:
        # Concrete subclasses (embeddings/rerank wrappers) declare buffer_tokens
        # as a pydantic Field; read it generically so the shared truncation
        # logic stays here without shadowing the subclass field.
        return int(getattr(self, "buffer_tokens", 0))

    def _truncate_text(
        self, text: str, max_chars: int | None, max_tokens: int | None, text_type: str
    ) -> str:
        if not max_chars and not max_tokens:
            return text

        final_text = text

        if max_tokens and self._token_counter is not None:
            effective_tokens = max_tokens - self._buffer_tokens
            truncated, token_count = self._token_counter.truncate_to_token_limit(
                text, effective_tokens
            )
            if len(truncated) < len(text):
                final_text = truncated
                original_count = self._token_counter.count_tokens(text)
                logger.warning(
                    "%s token count (%s) exceeds maximum (%s). Truncating.",
                    text_type.capitalize(),
                    original_count,
                    max_tokens,
                )

        if max_chars and len(text) > max_chars:
            char_truncated = text[:max_chars]
            if len(char_truncated) < len(final_text):
                final_text = char_truncated
                logger.warning(
                    "%s character count (%s) exceeds maximum (%s). Truncating.",
                    text_type.capitalize(),
                    len(text),
                    max_chars,
                )

        return final_text


class BaseBedrockModelFactory(Generic[ModelIdT, ModelInfoT, WrapperT], ABC):
    # Per-request socket read timeout. A single LLM/embedding response completes
    # in well under this; the cap exists so a stalled Converse call (no bytes
    # from the server) fails fast and is retried, instead of blocking a worker
    # for many minutes. Kept generous enough for long generations (community
    # reports, extended thinking) but short enough that a hung socket doesn't
    # freeze a whole ProcessPool-parallel stage (e.g. claim extraction).
    # A non-streaming call sends no bytes until the whole answer is ready, so
    # this also caps generation time. It sits above BatchProcessor's default
    # 300 s call timeout so that wall-clock limit, not a race with this one,
    # ends a long ingestion call: botocore retries a read timeout itself, and
    # at equal values that silent re-send started inside the call timeout's
    # window, only to be abandoned when it fired.
    BOTO_READ_TIMEOUT: ClassVar[int] = 330
    # TCP connect timeout. Without this, a client whose endpoint is unreachable
    # (e.g. a private/no-NAT VPC missing the relevant Bedrock interface endpoint,
    # such as bedrock-agent-runtime for the Rerank API) blocks at the socket
    # level effectively forever. A bounded connect timeout turns that hang into a
    # prompt, retryable error — which callers that degrade gracefully (e.g. the
    # hybrid scorer's rerank fallback to RRF-only) can actually catch.
    BOTO_CONNECT_TIMEOUT: ClassVar[int] = 10
    BOTO_MAX_ATTEMPTS: ClassVar[int] = 5
    # "adaptive" adds client-side rate limiting on top of retries, which is
    # materially better for throttling-heavy Bedrock workloads than "standard".
    BOTO_RETRY_MODE: ClassVar[Literal["legacy", "standard", "adaptive"]] = "adaptive"
    # urllib3 connection-pool bounds. botocore defaults to 10 connections per
    # client, but ingestion issues up to max_concurrency x chunk_concurrency
    # concurrent Bedrock calls through one client; a smaller pool discards and
    # re-opens connections ("Connection pool is full"), paying a TLS handshake
    # per call. The cap keeps an extreme concurrency setting from opening an
    # unbounded number of sockets.
    BOTO_MIN_POOL_CONNECTIONS: ClassVar[int] = 10
    BOTO_MAX_POOL_CONNECTIONS: ClassVar[int] = 200

    @classmethod
    def max_pool_connections(cls, config: Config) -> int:
        """urllib3 pool size for a Bedrock client under ``config``'s concurrency."""
        processing = config.processing
        wanted = processing.max_concurrency * processing.chunk_concurrency
        return max(
            cls.BOTO_MIN_POOL_CONNECTIONS, min(wanted, cls.BOTO_MAX_POOL_CONNECTIONS)
        )

    def _boto_config(self, read_timeout: int | None = None) -> BotoConfig:
        # botocore accepts a plain retries dict at runtime; its stub uses a
        # private _RetryDict that a local dict does not nominally satisfy.
        retries = {"max_attempts": self.BOTO_MAX_ATTEMPTS, "mode": self.BOTO_RETRY_MODE}
        if read_timeout is not None:
            return BotoConfig(
                connect_timeout=self.BOTO_CONNECT_TIMEOUT,
                read_timeout=read_timeout,
                retries=retries,  # type: ignore[arg-type]
                max_pool_connections=self.max_pool_connections(self.config),
            )
        return BotoConfig(
            connect_timeout=self.BOTO_CONNECT_TIMEOUT,
            retries=retries,  # type: ignore[arg-type]
            max_pool_connections=self.max_pool_connections(self.config),
        )

    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        region_name: str | None = None,
    ) -> None:
        self.config = config
        self.boto_session = boto_session or boto3.Session(
            profile_name=config.aws.profile_name
        )
        self.boto_session = get_assumed_role_boto_session(
            self.boto_session, assumed_role_arn=config.aws.bedrock.assumed_role_arn
        )
        self.region_name = region_name or config.aws.bedrock_region
        boto_config = self._boto_config(read_timeout=self.BOTO_READ_TIMEOUT)
        # Service name is resolved dynamically per subclass, so it is a plain
        # str and does not match types-boto3's literal-overloaded client().
        self._client = self.boto_session.client(
            self._get_boto_service_name(),  # type: ignore[call-overload]
            region_name=self.region_name,
            config=boto_config,
        )
        logger.debug(
            "Initialized %s for region: '%s'", self.__class__.__name__, self.region_name
        )

    @abstractmethod
    def _get_boto_service_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_model(self, model_id: ModelIdT, **kwargs: Any) -> WrapperT:
        raise NotImplementedError

    @abstractmethod
    def get_model_info(self, model_id: ModelIdT) -> ModelInfoT | None:
        raise NotImplementedError


class BedrockCrossRegionModelHelper:
    # Cache the system-defined inference-profile id set per region. Resolving a
    # model id used to call list_inference_profiles on EVERY get_model() (twice
    # when enable_global_profile) — once per model creation across the whole
    # pipeline. The set is process-stable, so fetch it once per region.
    _profiles_by_region: ClassVar[dict[str, set[str]]] = {}

    @staticmethod
    def get_cross_region_model_id(
        boto_session: boto3.Session,
        model_id: str,
        region_name: str,
        assumed_role_arn: str | None = None,
        enable_global_profile: bool = False,
    ) -> str:
        model_id = str(model_id)
        if base_model_id(model_id) != model_id:
            # Already an inference-profile id (e.g. 'us.anthropic.…').
            return model_id
        try:
            boto_session = get_assumed_role_boto_session(
                boto_session, assumed_role_arn=assumed_role_arn
            )
            bedrock_client = boto_session.client("bedrock", region_name=region_name)

            if enable_global_profile:
                global_model_id = (
                    BedrockCrossRegionModelHelper._build_cross_region_model_id(
                        model_id, region_name, is_global=True
                    )
                )
                if BedrockCrossRegionModelHelper._is_cross_region_model_available(
                    bedrock_client, global_model_id, region_name
                ):
                    logger.debug(
                        "Using global cross-region model: '%s'", global_model_id
                    )
                    return global_model_id
            regional_model_id = (
                BedrockCrossRegionModelHelper._build_cross_region_model_id(
                    model_id, region_name, is_global=False
                )
            )
            if BedrockCrossRegionModelHelper._is_cross_region_model_available(
                bedrock_client, regional_model_id, region_name
            ):
                logger.debug(
                    "Using regional cross-region model: '%s'", regional_model_id
                )
                return regional_model_id
            logger.debug(
                "Cross-region models not available, using standard model: '%s'",
                model_id,
            )
            return model_id
        except Exception as e:
            logger.warning(
                "Failed to resolve cross-region model for '%s': %s. Falling back to standard model.",
                model_id,
                e,
            )
            return model_id

    @staticmethod
    def _build_cross_region_model_id(
        model_id: str, region_name: str, is_global: bool = False
    ) -> str:
        if is_global:
            return f"global.{model_id}"
        prefix = "apac" if region_name.startswith("ap-") else region_name[:2]
        return f"{prefix}.{model_id}"

    @classmethod
    def _get_available_profiles(cls, bedrock_client: Any, region_name: str) -> set[str]:
        """System-defined inference-profile ids for a region, fetched once.

        Cached per region on the class so resolving many models in one process
        does not re-issue list_inference_profiles each time.
        """
        cached = cls._profiles_by_region.get(region_name)
        if cached is not None:
            return cached
        try:
            response = bedrock_client.list_inference_profiles(
                maxResults=1000, typeEquals="SYSTEM_DEFINED"
            )
            profiles = {
                profile["inferenceProfileId"]
                for profile in response.get("inferenceProfileSummaries", [])
            }
        except Exception as e:
            raise AWSServiceError(
                f"Failed to check cross-region model availability: {e}"
            ) from e
        cls._profiles_by_region[region_name] = profiles
        return profiles

    @classmethod
    def _is_cross_region_model_available(
        cls, bedrock_client: Any, cross_region_id: str, region_name: str
    ) -> bool:
        return cross_region_id in cls._get_available_profiles(
            bedrock_client, region_name
        )


class BedrockEmbeddingsWrapper(BaseBedrockWrapper, BedrockEmbeddings):
    buffer_tokens: int = Field(default=512, ge=0)
    max_sequence_length: int | None = Field(default=None)
    max_sequence_tokens: int | None = Field(default=None)
    # Application-level retry for transient Bedrock model errors (e.g. HTTP 424
    # ModelErrorException) that botocore's retry modes do not cover. Applied
    # per InvokeModel call, so every sync/async embed path benefits and a
    # failure mid-batch does not restart the already-embedded texts. The
    # factory sets it from aws.bedrock.transient_retry.
    transient_retry: TransientRetryConfig = Field(default_factory=TransientRetryConfig)

    def _invoke_model(self, input_body: dict[str, Any] | None = None) -> dict[str, Any]:
        # Single choke point for every embedding request in langchain-aws
        # (embed_documents, embed_query, Cohere batch, and the async variants,
        # which run embed_query in an executor).
        body = input_body or {}
        return call_with_transient_retry(
            lambda: super(BedrockEmbeddingsWrapper, self)._invoke_model(
                input_body=body
            ),
            operation=f"embed:{self.model_id}",
            policy=self.transient_retry,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            logger.warning("No texts provided for embedding")
            return []
        truncated_texts = [
            self._truncate_text(
                text, self.max_sequence_length, self.max_sequence_tokens, "document"
            )
            for text in texts
        ]
        return super().embed_documents(truncated_texts)

    def embed_query(self, text: str) -> list[float]:
        truncated_text = self._truncate_text(
            text, self.max_sequence_length, self.max_sequence_tokens, "query"
        )
        return super().embed_query(truncated_text)


class BedrockEmbeddingModelFactory(
    BaseBedrockModelFactory[
        EmbeddingModelId, EmbeddingModelInfo, BedrockEmbeddingsWrapper
    ]
):
    def _get_boto_service_name(self) -> str:
        return "bedrock-runtime"

    def get_model_info(self, model_id: EmbeddingModelId) -> EmbeddingModelInfo | None:
        return _EMBEDDING_MODEL_INFO.get(model_id)

    def get_model(
        self, model_id: EmbeddingModelId, **kwargs: Any
    ) -> BedrockEmbeddingsWrapper:
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise EmbeddingModelError(
                f"Unsupported embedding model ID: '{model_id.value}'"
            )

        model_kwargs = {}
        dimensions = kwargs.pop("dimensions", None)

        if dimensions:
            supported_dims = model_info.dimensions
            is_supported = False
            if isinstance(supported_dims, list):
                is_supported = dimensions in supported_dims
            elif isinstance(supported_dims, int):
                is_supported = dimensions == supported_dims

            if not is_supported:
                raise EmbeddingModelError(
                    f"Dimension {dimensions} is not supported by model '{model_id.value}'. "
                    f"Supported dimensions: {supported_dims}"
                )
            if isinstance(supported_dims, list):
                model_kwargs["dimensions"] = dimensions

        # CountTokens takes a Converse-shaped input, which embedding models do
        # not accept: without a client the counter uses the script-aware
        # estimate instead of paying a failing round trip per text.
        token_counter = BedrockTokenCounter(model_id=model_id.value, client=None)
        model = BedrockEmbeddingsWrapper(
            client=self._client,
            model_id=model_id.value,
            model_kwargs=model_kwargs,
            max_sequence_length=model_info.max_sequence_length,
            max_sequence_tokens=model_info.max_sequence_tokens,
            transient_retry=self.config.aws.bedrock.transient_retry,
            # Accepted by BaseBedrockWrapper.__init__; pydantic's generated
            # __init__ signature hides it from mypy.
            token_counter=token_counter,  # type: ignore[call-arg]
            **kwargs,
        )
        logger.debug("Created embedding model: '%s'", model_id.value)
        return model


_GUARDRAIL_INTERVENED_STOP_REASON = "guardrail_intervened"
# InvokeModel response-body flag (langchain_aws GUARDRAILS_BODY_KEY).
_GUARDRAIL_ACTION_KEY = "amazon-bedrock-guardrailAction"


def _message_guardrail_intervened(metadata: dict[str, Any]) -> bool:
    """Best-effort detection of a guardrail intervention in response metadata.

    Converse (``ChatBedrockConverse``) reports ``stopReason ==
    "guardrail_intervened"``; InvokeModel surfaces the body flag
    ``amazon-bedrock-guardrailAction == "INTERVENED"`` when langchain_aws
    propagates it. Anything else is treated as "not intervened".
    """
    stop_reason = metadata.get("stopReason") or metadata.get("stop_reason")
    if stop_reason == _GUARDRAIL_INTERVENED_STOP_REASON:
        return True
    return metadata.get(_GUARDRAIL_ACTION_KEY) == "INTERVENED"


class GuardrailInterventionHandler(BaseCallbackHandler):
    """Make guardrail interventions visible instead of silently degrading.

    When a guardrail blocks or masks a call, Bedrock still returns a normal
    response (the configured "blocked" message), so a downstream parser just
    sees text with no entities/answer. This handler logs each intervention at
    WARNING with the model purpose and a process-wide running count.

    It deliberately does NOT raise: on the query path the blocked message *is*
    the intended user-facing response, and raising from a callback would turn
    it into an opaque chain error. Operators alert on the WARNING.
    """

    _count: ClassVar[int] = 0
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, guardrail_identifier: str, purpose: ModelPurpose) -> None:
        self.guardrail_identifier = guardrail_identifier
        self.purpose = purpose
        # InvokeModel with trace enabled reports through on_llm_error and then
        # still calls on_llm_end; remember those runs to count each call once.
        self._flagged_runs: set[UUID] = set()

    def _record(self) -> None:
        with self._lock:
            type(self)._count += 1
            total = type(self)._count
        logger.warning(
            "Bedrock guardrail '%s' intervened on a %s model call; the response "
            "was blocked or masked (interventions in this process: %d)",
            self.guardrail_identifier,
            self.purpose.value,
            total,
        )

    def on_llm_error(
        self, error: BaseException, *, run_id: UUID, **kwargs: Any
    ) -> None:
        if kwargs.get("reason") == "GUARDRAIL_INTERVENED":
            self._flagged_runs.add(run_id)
            self._record()

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        if run_id in self._flagged_runs:
            self._flagged_runs.discard(run_id)
            return
        for generations in response.generations:
            for generation in generations:
                if isinstance(
                    generation, ChatGeneration
                ) and _message_guardrail_intervened(
                    generation.message.response_metadata
                ):
                    self._record()
                    return


class BedrockLanguageModelFactory(
    BaseBedrockModelFactory[str, LanguageModelInfo, ChatBedrock | ChatBedrockConverse]
):
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.0
    DEFAULT_TOP_K: ClassVar[int] = 50
    DEFAULT_THINKING_BUDGET_TOKENS: ClassVar[int] = 2048

    def _get_boto_service_name(self) -> str:
        return "bedrock-runtime"

    def get_model_info(self, model_id: str) -> LanguageModelInfo:
        """Capability record for any model id (see get_language_model_info)."""
        return get_language_model_info(
            model_id, self.config.aws.bedrock.model_overrides
        )

    def get_model(
        self,
        model_id: str,
        **kwargs: Any,
    ) -> ChatBedrock | ChatBedrockConverse:
        model_id = str(model_id)
        model_info = self.get_model_info(model_id)
        resolved_model_id = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            self.boto_session,
            model_id,
            self.region_name or "",
            assumed_role_arn=self.config.aws.bedrock.assumed_role_arn,
            enable_global_profile=self.config.aws.bedrock.enable_global_profile,
        )
        is_cross_region = base_model_id(resolved_model_id) != resolved_model_id
        if model_info.requires_inference_profile and not is_cross_region:
            # Claude 4.6+ ships INFERENCE_PROFILE-only (no ON_DEMAND throughput),
            # so the bare model id is not invocable. Resolution silently falls
            # back to it, which would surface later as an opaque Bedrock error —
            # fail here with the actual remedy instead.
            raise LanguageModelError(
                f"Model '{model_id}' is only available through a "
                f"cross-region inference profile, but none resolved in region "
                f"'{self.region_name}'. Enable aws.bedrock.enable_global_profile, "
                f"grant bedrock:ListInferenceProfiles, or choose a region where "
                f"a profile for this model exists."
            )
        # ChatBedrock speaks the Anthropic InvokeModel body; every other provider
        # (and every inference profile) goes through the provider-neutral
        # Converse API, so an on-demand non-Anthropic model is never sent an
        # Anthropic-shaped body.
        use_converse = is_cross_region or model_info.provider != "anthropic"
        # The tier picks the configured reasoning effort (fast vs default); a
        # caller may pass model_tier explicitly.
        kwargs.setdefault("model_tier", self.config.aws.bedrock.model_tier(model_id))
        model_config = self._build_model_config(
            model_info, resolved_model_id, use_converse, **kwargs
        )
        model_class = ChatBedrockConverse if use_converse else ChatBedrock
        model = model_class(**model_config)
        logger.debug(
            "Created language model: '%s' with class %s",
            resolved_model_id,
            model_class.__name__,
        )
        return model

    def _build_model_config(
        self,
        model_info: LanguageModelInfo,
        resolved_model_id: str,
        is_cross_region: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        enable_thinking = kwargs.get("enable_thinking", False)
        # An explicit max_tokens is clamped as requested; otherwise the default
        # cap applies, raised to the prompt's own output floor.
        final_max_tokens = self._validate_max_tokens(
            kwargs.get("max_tokens")
            or effective_max_output_tokens(
                model_info,
                self.config.aws.bedrock,
                kwargs.get("min_output_tokens", 0),
            ),
            model_info,
        )
        config = self._build_base_config(
            resolved_model_id, is_cross_region, model_info, **kwargs
        )
        token_params: dict[str, Any] = {"max_tokens": final_max_tokens}
        if model_info.supports_sampling_params:
            temperature = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)
            # Thinking models sample at a fixed temperature of 1.0.
            final_temperature = (
                1.0
                if self._should_enable_thinking(enable_thinking, model_info)
                else temperature
            )
            if final_temperature != temperature:
                logger.debug("Adjusting temperature to 1.0 for thinking mode")
            token_params["temperature"] = final_temperature
        else:
            # Claude 4.7+ removed temperature/top_p/top_k; sending any of them
            # returns a 400. Behaviour is steered by prompting and 'effort'.
            logger.debug(
                "Omitting sampling parameters for '%s' (unsupported by model)",
                resolved_model_id,
            )
        if is_cross_region:
            config.update(token_params)
        else:
            config["model_kwargs"].update(token_params)
        if model_info.native_1m_context_window:
            # 1M is the default window on Claude 5 — the beta opt-in header
            # that older models need is unnecessary (and misleading) here.
            logger.debug(
                "Model '%s' has a native 1M context window; skipping beta header",
                resolved_model_id,
            )
        elif (
            self.config.aws.bedrock.enable_1m_context
            and model_info.supports_1m_context_window
            and model_info.provider == "anthropic"
        ):
            if is_cross_region:
                config.setdefault("additional_model_request_fields", {}).update(
                    {"anthropic_beta": ["context-1m-2025-08-07"]}
                )
            else:
                config["model_kwargs"].setdefault(
                    "additionalModelRequestFields", {}
                ).update({"anthropic_beta": ["context-1m-2025-08-07"]})
            logger.debug("Applied 1M context window support")
        self._apply_model_features(config, model_info, is_cross_region, **kwargs)
        return config

    def _build_base_config(
        self,
        resolved_model_id: str,
        is_cross_region: bool,
        model_info: LanguageModelInfo,
        **kwargs: Any,
    ) -> dict[str, Any]:
        config = {
            "model_id": resolved_model_id,
            "region_name": self.region_name,
            "client": self._client,
            "callbacks": kwargs.get("callbacks"),
        }
        if (
            self.boto_session.profile_name
            and self.boto_session.profile_name != "default"
        ):
            config["credentials_profile_name"] = self.boto_session.profile_name
        # No stop sequence: both Converse and the InvokeModel Messages body are
        # turn-structured, so the legacy "\n\nHuman:" text-completion marker
        # only cut off answers whose text contains it (chat transcripts).
        if is_cross_region and model_info.supports_streaming is not None:
            config["disable_streaming"] = not model_info.supports_streaming
        if not is_cross_region:
            # top_k is a sampling parameter; Claude 4.7+ rejects it.
            config["model_kwargs"] = (
                {"top_k": kwargs.get("top_k", self.DEFAULT_TOP_K)}
                if model_info.supports_sampling_params
                else {}
            )
        return config

    def _apply_model_features(
        self,
        config: dict[str, Any],
        model_info: LanguageModelInfo,
        is_cross_region: bool,
        **kwargs: Any,
    ) -> None:
        enable_think = kwargs.get("enable_thinking", False)
        if self._should_enable_thinking(enable_think, model_info):
            think_config = self._build_thinking_config(model_info, **kwargs)
            if is_cross_region:
                config.setdefault("additional_model_request_fields", {}).update(
                    think_config
                )
            else:
                config.setdefault("model_kwargs", {}).update(think_config)
        self._apply_guardrail(
            config,
            is_cross_region,
            ModelPurpose(kwargs.get("model_purpose", ModelPurpose.QUERY)),
        )

    def _build_thinking_config(
        self, model_info: LanguageModelInfo, **kwargs: Any
    ) -> dict[str, Any]:
        """Assemble the provider-specific reasoning request fields for a model.

        - OpenAI GPT: ``{"reasoning": {"effort": ...}}`` in
          ``additionalModelRequestFields``. Bedrock forwards it to the model
          unchanged; the flat Chat Completions ``reasoning_effort`` is rejected
          there as an unknown parameter.
        - Anthropic adaptive (Claude 4.6+): ``{"type": "adaptive"}`` plus
          ``effort`` in its own ``output_config`` object — nesting it inside
          ``thinking`` raises a ``ValidationException``. Adaptive-only models
          (Claude 4.7+) reject the manual ``budget_tokens`` shape with a 400.
        - Older Anthropic: ``{"type": "enabled", "budget_tokens": N}``.
        """
        if model_info.provider == "openai":
            effort = self._resolve_effort(model_info, **kwargs)
            logger.debug("Applied OpenAI reasoning (reasoning.effort='%s')", effort)
            return {"reasoning": {"effort": effort}}
        if not model_info.uses_adaptive_thinking:
            budget = kwargs.get(
                "thinking_budget_tokens", self.DEFAULT_THINKING_BUDGET_TOKENS
            )
            logger.debug("Applied extended thinking (budget_tokens=%d)", budget)
            return {"thinking": {"type": "enabled", "budget_tokens": budget}}
        effort = self._resolve_effort(model_info, **kwargs)
        logger.debug("Applied adaptive thinking (effort='%s')", effort)
        return {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort},
        }

    def _resolve_effort(self, model_info: LanguageModelInfo, **kwargs: Any) -> str:
        """Pick the effort level and validate it.

        A per-call ``effort`` wins; otherwise the configured effort of the
        call's ``model_tier`` (set by ``get_model`` from the model id, default
        tier when absent). Fails fast with the documented levels rather than
        letting Bedrock answer with a 400 on the first request.
        """
        effort = kwargs.get("effort")
        source = "the per-call effort"
        if not effort:
            tier: ModelTier = (
                "fast" if kwargs.get("model_tier") == "fast" else "default"
            )
            effort = self.config.aws.bedrock.tier_effort(tier)
            source = f"aws.bedrock.{tier}_effort"
        if effort not in EFFORT_LEVELS:
            raise LanguageModelError(
                f"Invalid effort level '{effort}' ({source}). "
                f"Valid levels: {sorted(EFFORT_LEVELS)}"
            )
        allowed = model_info.supported_efforts
        if allowed is not None and effort not in allowed:
            raise LanguageModelError(
                f"Effort level '{effort}' ({source}) is not supported by this "
                f"model. Supported levels: {sorted(allowed)}"
            )
        return str(effort)

    def _apply_guardrail(
        self,
        config: dict[str, Any],
        is_cross_region: bool,
        purpose: ModelPurpose = ModelPurpose.QUERY,
    ) -> None:
        """Attach Bedrock Guardrails to the model when configured.

        ChatBedrockConverse exposes ``guardrail_config`` (Converse-API shape);
        ChatBedrock exposes ``guardrails`` (InvokeModel shape). When no guardrail
        identifier is set the model is created without guardrails (no-op).
        ``purpose`` scopes it via ``guardrail.apply_to``: by default only
        query-path models are guarded. A guarded model also gets a
        :class:`GuardrailInterventionHandler` so interventions are logged.
        """
        guardrail = self.config.aws.bedrock.guardrail
        if guardrail.identifier is None or not guardrail.applies_to(purpose):
            if guardrail.enabled:
                logger.debug(
                    "Skipping Bedrock guardrail for %s model (apply_to='%s')",
                    purpose.value,
                    guardrail.apply_to,
                )
            return
        if is_cross_region:
            # ChatBedrockConverse passes guardrail_config straight to the
            # Converse API, whose trace field is the literal "enabled"/"disabled".
            config["guardrail_config"] = {
                "guardrailIdentifier": guardrail.identifier,
                "guardrailVersion": guardrail.version,
                "trace": "enabled" if guardrail.trace else "disabled",
            }
        else:
            # ChatBedrock (InvokeModel) treats guardrails["trace"] as a
            # truthiness flag (`if self.guardrails.get("trace")`), so a non-empty
            # string like "disabled" would wrongly enable tracing — pass a bool.
            config["guardrails"] = {
                "guardrailIdentifier": guardrail.identifier,
                "guardrailVersion": guardrail.version,
                "trace": guardrail.trace,
            }
        handler = GuardrailInterventionHandler(guardrail.identifier, purpose)
        callbacks = config.get("callbacks")
        if callbacks is None:
            config["callbacks"] = [handler]
        elif isinstance(callbacks, BaseCallbackManager):
            # Copy first: the manager is caller-owned and may be shared by other
            # (unguarded) models, which must not start counting interventions.
            manager = callbacks.copy()
            manager.add_handler(handler, inherit=False)
            config["callbacks"] = manager
        else:
            config["callbacks"] = [*callbacks, handler]
        logger.debug(
            "Applied Bedrock guardrail '%s' to %s model",
            guardrail.identifier,
            purpose.value,
        )

    @staticmethod
    def _validate_max_tokens(max_tokens: int, model_info: LanguageModelInfo) -> int:
        if max_tokens > model_info.max_output_tokens:
            logger.warning(
                "Requested max_tokens (%d) exceeds model's maximum (%d). Adjusting.",
                max_tokens,
                model_info.max_output_tokens,
            )
            return model_info.max_output_tokens
        return max_tokens

    @staticmethod
    def _should_enable_thinking(enable: bool, model_info: LanguageModelInfo) -> bool:
        if not model_info.supports_thinking:
            return False
        # Adaptive-only Claude models think by default (and Sonnet 5 / 5.5 and
        # Opus 5.5 reject {'type': 'disabled'} outright) and GPT models always
        # reason, so their requests carry the reasoning config regardless of
        # the caller's flag — that block is also what carries the configured
        # effort level, which would otherwise be silently dropped.
        return enable or model_info.always_reasons


class BedrockRerankWrapper(BaseBedrockWrapper, BedrockRerank):
    buffer_tokens: int = Field(default=64, ge=0)
    max_documents: int = Field(default=1000, ge=1)
    max_query_length: int | None = Field(default=None)
    max_query_tokens: int | None = Field(default=None)
    max_document_length: int | None = Field(default=None)
    max_document_tokens: int | None = Field(default=None)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: list[BaseCallbackHandler] | BaseCallbackManager | None = None,
    ) -> list[Document]:
        if len(documents) > self.max_documents:
            logger.warning(
                "Document count (%s) exceeds limit (%s). Using first %s documents.",
                len(documents),
                self.max_documents,
                self.max_documents,
            )
            documents = documents[: self.max_documents]

        if self.top_n is not None and len(documents) < self.top_n:
            # Clamp on a copy: this model is shared by concurrent queries, so
            # mutating its top_n (even temporarily) would race.
            logger.info(
                "Adjusted top_n from %s to %s to match document count",
                self.top_n,
                len(documents),
            )
            clamped = self.model_copy(update={"top_n": len(documents)})
            return clamped.compress_documents(documents, query, callbacks=callbacks)

        truncated_query = self._truncate_text(
            query, self.max_query_length, self.max_query_tokens, "query"
        )

        for doc in documents:
            doc.page_content = self._truncate_text(
                doc.page_content,
                self.max_document_length,
                self.max_document_tokens,
                "document",
            )

        try:
            result = super().compress_documents(
                documents, truncated_query, callbacks=callbacks
            )
            return list(result)
        except Exception as e:
            raise RerankModelError(f"Reranking failed: {e}") from e


class BedrockRerankModelFactory(
    BaseBedrockModelFactory[str, RerankModelInfo, BedrockRerankWrapper]
):
    DEFAULT_TOP_K: ClassVar[int] = 100

    def _get_boto_service_name(self) -> str:
        return "bedrock-agent-runtime"

    def get_model_info(self, model_id: str) -> RerankModelInfo | None:
        return _RERANK_MODEL_INFO.get(model_id)

    def get_model(
        self, model_id: RerankModelId | str, **kwargs: Any
    ) -> BedrockRerankWrapper:
        if isinstance(model_id, str):
            try:
                model_id = RerankModelId(model_id)
            except ValueError as e:
                raise RerankModelError(
                    f"Unsupported rerank model ID: '{model_id}'"
                ) from e

        model_info = self.get_model_info(model_id)
        if not model_info:
            raise RerankModelError(f"Unsupported rerank model ID: '{model_id.value}'")

        top_k = kwargs.pop("top_k", self.DEFAULT_TOP_K)
        if top_k > model_info.max_documents:
            logger.warning(
                "Requested 'top_k' (%s) exceeds model's maximum (%s). Adjusting.",
                top_k,
                model_info.max_documents,
            )
            top_k = model_info.max_documents

        model_arn = (
            f"arn:aws:bedrock:{self.region_name}::foundation-model/{model_id.value}"
        )

        # Rerank models do not accept CountTokens' Converse input either, so
        # the counter gets no client and uses the script-aware estimate.
        token_counter = BedrockTokenCounter(model_id=model_id.value, client=None)
        model = BedrockRerankWrapper(
            model_arn=model_arn,
            top_n=top_k,
            max_query_length=model_info.max_query_length,
            max_query_tokens=model_info.max_query_tokens,
            max_document_length=model_info.max_document_length,
            max_document_tokens=model_info.max_document_tokens,
            region_name=self.region_name,
            credentials_profile_name=self.boto_session.profile_name,
            client=self._client,
            # Accepted by BaseBedrockWrapper.__init__; pydantic's generated
            # __init__ signature hides it from mypy.
            token_counter=token_counter,  # type: ignore[call-arg]
            **kwargs,
        )
        logger.debug("Created rerank model: '%s'", model_id.value)
        return model


def get_assumed_role_boto_session(
    boto_session: boto3.Session,
    assumed_role_arn: str | None = None,
    role_session_name: str = DEFAULT_ROLE_SESSION_NAME,
    duration_seconds: int = 3600,
) -> boto3.Session:
    if assumed_role_arn is None:
        return boto_session

    try:
        credentials = boto_session.get_credentials()
        if credentials and hasattr(credentials, "method"):
            if credentials.method == "assume-role":
                sts_client = boto_session.client("sts")
                try:
                    caller_identity = sts_client.get_caller_identity()
                    current_arn = caller_identity.get("Arn", "")
                    if "assumed-role" in current_arn:
                        current_role_name = (
                            current_arn.split("/")[-2] if "/" in current_arn else ""
                        )
                        target_role_name = (
                            assumed_role_arn.split("/")[-1]
                            if "/" in assumed_role_arn
                            else ""
                        )

                        if current_role_name == target_role_name:
                            logger.debug(
                                "Already using assumed role '%s', skipping duplicate assume",
                                assumed_role_arn,
                            )
                            return boto_session
                except Exception as e:
                    logger.debug("Could not verify current role identity: %s", e)
    except Exception as e:
        logger.debug("Could not check assumed role status: %s", e)

    logger.info(
        "Using aws-assume-role-lib to assume role: '%s' with session name: '%s'",
        assumed_role_arn,
        role_session_name,
    )
    return assume_role(
        boto_session,
        assumed_role_arn,
        RoleSessionName=role_session_name,
        DurationSeconds=duration_seconds,
    )
