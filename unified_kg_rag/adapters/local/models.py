# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Language-model factories for OpenAI-compatible servers and Ollama.

Both conform to ``LLMFactoryPort``: ``get_model`` returns a LangChain chat
model and ``get_model_info`` a capability record built from
``providers.llm`` (window, output ceiling; no prompt caching, reasoning
controls or CountTokens). They accept and ignore the Bedrock-only keyword
arguments call sites pass (``enable_thinking``, ``model_tier``, ``top_k`` ...),
and attach the GenAI tracing callback when observability is enabled, as the
Bedrock factory does.

The LangChain integrations (``langchain-openai`` / ``langchain-ollama``, the
``[local]`` extra) are imported when a model is built, so selecting another
provider never needs them.
"""

from __future__ import annotations

import importlib
from typing import Any, ClassVar

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel

from unified_kg_rag.adapters.llm.retry import looks_transient
from unified_kg_rag.domain.models import Config, ModelPurpose
from unified_kg_rag.domain.models.config import TransientRetryConfig
from unified_kg_rag.domain.models.provider_config import (
    EndpointConfig,
    LLMProviderConfig,
)
from unified_kg_rag.ports.model_factory import (
    LanguageModelInfo,
    effective_max_output_tokens,
)
from unified_kg_rag.shared import ConfigurationError, get_logger, observability

logger = get_logger(__name__)

# Sent when an OpenAI-compatible server needs no key: the client refuses to
# start without one (or OPENAI_API_KEY).
PLACEHOLDER_API_KEY = "not-needed"


def import_optional(module: str, kind: str) -> Any:
    """Import a ``[local]``-extra module, failing with the remedy if absent."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ConfigurationError(
            f"Provider kind '{kind}' needs '{module}', which is not installed. "
            "Install the local extra: uv sync --extra local (or pip install -e "
            "'.[local]' in a source checkout)."
        ) from e


def api_key_value(endpoint: EndpointConfig) -> str:
    """The configured API key, or a placeholder for servers that need none."""
    if endpoint.api_key is None:
        return PLACEHOLDER_API_KEY
    return endpoint.api_key.get_secret_value()


def _tracing_callbacks(
    callbacks: Any,
    model_id: str,
    provider_name: str,
    purpose: ModelPurpose,
    max_tokens: int,
    temperature: float,
) -> Any:
    """``callbacks`` plus the GenAI tracing handler when tracing is enabled."""
    if not observability.is_enabled():
        return callbacks
    from unified_kg_rag.adapters.observability.langchain_callback import (
        GenAITracingCallbackHandler,
    )
    from unified_kg_rag.shared.utils.langchain import with_callback

    handler: BaseCallbackHandler = GenAITracingCallbackHandler(
        model_id,
        provider_name=provider_name,
        purpose=purpose.value,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return with_callback(callbacks, handler)


def local_transient_retry(endpoint: EndpointConfig) -> TransientRetryConfig:
    """Retry policy for a client that does not retry on its own (Ollama).

    ``endpoint.max_attempts`` attempts with the default backoff and wall-clock
    budget of :class:`TransientRetryConfig`.
    """
    return TransientRetryConfig(max_attempts=endpoint.max_attempts)


class _LocalLanguageModelFactory:
    """Shared request shaping; subclasses build the provider's chat model."""

    provider_name: ClassVar[str]
    kind: ClassVar[str]

    def __init__(self, config: Config, *, transport: Any = None) -> None:
        """Args:
        config: Framework config; ``providers.llm`` configures the server.
        transport: Optional ``httpx`` transport for every client the factory
            builds (tests use ``httpx.MockTransport``; no network).
        """
        self.config = config
        self.settings: LLMProviderConfig = config.providers.llm
        self.transport = transport

    @staticmethod
    def is_transient_error(exc: BaseException) -> bool:
        """Whether ``exc`` is worth a retry (the ingestion batch retries).

        Connect/read timeouts and dropped connections (httpx, the OpenAI
        client) and HTTP 408/429/5xx (``openai.APIStatusError``,
        ``ollama.ResponseError``). The ingestion batch retries use it on both
        servers, and ``setup_chain`` the query-chain retry on Ollama
        (``transient_retry``).
        """
        return looks_transient(exc)

    def get_model_info(self, model_id: str) -> LanguageModelInfo:
        """Capability record of every model on this server (from config)."""
        return LanguageModelInfo(
            provider="other",
            context_window_size=self.settings.context_window_size,
            max_output_tokens=self.settings.max_output_tokens,
            supports_prompt_caching=False,
            supports_thinking=False,
            supports_sampling_params=True,
            supports_count_tokens=False,
        )

    def get_model(self, model_id: str, **kwargs: Any) -> BaseChatModel:
        model_id = str(model_id)
        info = self.get_model_info(model_id)
        max_tokens = min(
            int(
                kwargs.get("max_tokens")
                or effective_max_output_tokens(
                    info,
                    self.config.aws.bedrock,
                    kwargs.get("min_output_tokens") or 0,
                )
            ),
            info.max_output_tokens,
        )
        temperature = kwargs.get("temperature", self.settings.temperature)
        purpose = ModelPurpose(kwargs.get("model_purpose", ModelPurpose.QUERY))
        callbacks = _tracing_callbacks(
            kwargs.get("callbacks"),
            model_id,
            self.provider_name,
            purpose,
            max_tokens,
            temperature,
        )
        model = self._build(model_id, max_tokens, temperature, callbacks)
        logger.debug(
            "Created %s language model '%s' (max_tokens %s)",
            self.kind,
            model_id,
            max_tokens,
        )
        return model

    def _build(
        self, model_id: str, max_tokens: int, temperature: float, callbacks: Any
    ) -> BaseChatModel:
        raise NotImplementedError


class OpenAICompatibleLanguageModelFactory(_LocalLanguageModelFactory):
    """Chat models on an OpenAI-compatible server (``ChatOpenAI``).

    ``providers.llm.base_url`` points at the server's ``/v1`` root (vLLM,
    llama.cpp server, LM Studio, or OpenAI itself when unset).

    The OpenAI client retries transient errors itself
    (``providers.llm.max_attempts``), so this factory defines no
    ``transient_retry`` and query chains are not wrapped in a second retry.
    """

    provider_name = "openai"
    kind = "openai_compatible"

    def _build(
        self, model_id: str, max_tokens: int, temperature: float, callbacks: Any
    ) -> BaseChatModel:
        module = import_optional("langchain_openai", self.kind)
        extra: dict[str, Any] = {}
        if self.transport is not None:
            httpx = importlib.import_module("httpx")
            timeout = self.settings.timeout_seconds
            extra["http_client"] = httpx.Client(
                transport=self.transport, timeout=timeout
            )
            extra["http_async_client"] = httpx.AsyncClient(
                transport=self.transport, timeout=timeout
            )
        model: BaseChatModel = module.ChatOpenAI(
            model=model_id,
            base_url=self.settings.base_url,
            api_key=api_key_value(self.settings),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.settings.timeout_seconds,
            # The OpenAI client counts retries after the first attempt.
            max_retries=self.settings.max_attempts - 1,
            callbacks=callbacks,
            **extra,
        )
        return model


class OllamaLanguageModelFactory(_LocalLanguageModelFactory):
    """Chat models on an Ollama server (``ChatOllama``).

    ``providers.llm.context_window_size`` is sent as ``num_ctx``: Ollama's own
    default window is far smaller than the prompts the pipeline sends, and it
    truncates the prompt silently when it does not fit.

    The Ollama client never retries, so ``transient_retry`` (from
    ``providers.llm.max_attempts``) gives query chains the chain-level retry
    of ``setup_chain``; ingestion chains keep the batch retries.
    """

    provider_name = "ollama"
    kind = "ollama"

    @property
    def transient_retry(self) -> TransientRetryConfig:
        """The retry policy ``setup_chain`` applies to query chains."""
        return local_transient_retry(self.settings)

    def _build(
        self, model_id: str, max_tokens: int, temperature: float, callbacks: Any
    ) -> BaseChatModel:
        module = import_optional("langchain_ollama", self.kind)
        client_kwargs: dict[str, Any] = {"timeout": self.settings.timeout_seconds}
        if self.transport is not None:
            client_kwargs["transport"] = self.transport
        model: BaseChatModel = module.ChatOllama(
            model=model_id,
            base_url=self.settings.base_url,
            temperature=temperature,
            num_predict=max_tokens,
            num_ctx=self.settings.context_window_size,
            client_kwargs=client_kwargs,
            callbacks=callbacks,
        )
        return model
