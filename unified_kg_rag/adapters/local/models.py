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

import asyncio
import functools
import importlib
import threading
from collections.abc import Callable
from typing import Any, ClassVar

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel

from unified_kg_rag.adapters.llm.retry import looks_like_item_error, looks_transient
from unified_kg_rag.domain.models import Config, ModelPurpose
from unified_kg_rag.domain.models.config import TransientRetryConfig
from unified_kg_rag.domain.models.provider_config import (
    EndpointConfig,
    LLMProviderConfig,
)
from unified_kg_rag.ports.model_factory import (
    LanguageModelInfo,
    ceiled_output_tokens,
    effective_max_output_tokens,
)
from unified_kg_rag.shared import ConfigurationError, get_logger, observability
from unified_kg_rag.shared.extras import extra_hint

logger = get_logger(__name__)

# Sent when an OpenAI-compatible server needs no key: the client refuses to
# start without one (or OPENAI_API_KEY).
PLACEHOLDER_API_KEY = "not-needed"  # pragma: allowlist secret


def import_optional(module: str, kind: str, extra: str = "local") -> Any:
    """Import a module of an optional extra (``[local]`` unless ``extra``
    names another), failing with the remedy if absent."""
    try:
        # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import - literal module names
        return importlib.import_module(module)
    except ImportError as e:
        raise ConfigurationError(
            f"Provider kind '{kind}' needs '{module}', which is not installed. "
            f"Install {extra_hint(extra)}."
        ) from e


# httpx's default connection limit; only keep-alive is turned off.
_MAX_CONNECTIONS = 100


def _no_keepalive_limits() -> Any:
    """``httpx.Limits`` that keep no idle connection for reuse.

    A pooled keep-alive connection belongs to the event loop that opened it,
    and nothing closes it once that loop is closed (``asyncio.run`` closes its
    loop, not the clients used on it): every loop would leave a client and its
    sockets behind. Without keep-alive each connection closes on its own loop
    when its response is read, so a loop's client holds no socket afterwards.
    A model answers far slower than a connection opens.
    """
    httpx = importlib.import_module("httpx")
    return httpx.Limits(max_connections=_MAX_CONNECTIONS, max_keepalive_connections=0)


@functools.cache
def tls_context() -> Any:
    """The one TLS context of every model-server client in the process.

    httpx builds a context per client otherwise, loading the CA bundle
    (about 28 ms), and a query builds clients per model and per event loop.
    Built as httpx builds its default (``SSL_CERT_FILE`` or
    ``SSL_CERT_DIR`` when set, else certifi's bundle), when first needed.
    """
    httpx = importlib.import_module("httpx")
    return httpx.create_ssl_context()


def ollama_client_kwargs(timeout: float) -> dict[str, Any]:
    """``client_kwargs`` for an Ollama model: the request ``timeout`` and
    the shared TLS context (:func:`tls_context`)."""
    return {"timeout": timeout, "verify": tls_context()}


def ollama_async_client_kwargs(timeout: float | None = None) -> dict[str, Any]:
    """``async_client_kwargs`` for an Ollama model: no keep-alive connections,
    the shared TLS context, and the request ``timeout`` when one is given (a
    query call's)."""
    kwargs: dict[str, Any] = {
        "limits": _no_keepalive_limits(),
        "verify": tls_context(),
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    return kwargs


def request_timeout(config: Config, configured: float, purpose: ModelPurpose) -> float:
    """The request timeout of a model-server call made for ``purpose``: the
    provider's ``configured`` one, capped for query calls at
    ``search.model_timeout_seconds``."""
    if purpose is ModelPurpose.QUERY:
        return config.search.query_model_timeout(configured)
    return configured


class PerLoopAsyncClient:
    """Stands in for an async HTTP client, with one client per event loop.

    ``ChatOllama`` and ``OllamaEmbeddings`` build one async client at
    construction, and the OpenAI models keep one too; the
    connections belong to the event loop that opened them, so the same model
    used again from another loop (a second ``asyncio.run``, or the sync API's
    loop and then the caller's) fails with "Event loop is closed". The models
    are built once and reused, as the query chains are; this proxy gives
    every running loop its own client, built by ``build``, and drops the
    clients of closed loops whenever a client is looked up. The clients are
    built without keep-alive (``_no_keepalive_limits``), so a dropped client
    holds no socket.
    """

    def __init__(self, build: Callable[[], Any]) -> None:
        self._build = build
        self._clients: dict[asyncio.AbstractEventLoop, Any] = {}
        self._lock = threading.Lock()

    def _current(self) -> Any:
        loop = asyncio.get_running_loop()
        with self._lock:
            for closed in [other for other in self._clients if other.is_closed()]:
                del self._clients[closed]
            client = self._clients.get(loop)
            if client is None:
                client = self._clients[loop] = self._build()
        return client

    def __getattr__(self, name: str) -> Any:
        return getattr(self._current(), name)


def per_loop_async_client(model: Any, rebuild: Callable[[], Any]) -> Any:
    """Give ``model`` (an Ollama LangChain model) a per-loop async client.

    ``rebuild`` returns a new model with the same settings, built with
    ``async_client_kwargs=ollama_async_client_kwargs()``; its async client is
    the one for a new loop, so the client is built exactly as the library
    builds it (host, auth from the URL, client kwargs).
    """
    model._async_client = PerLoopAsyncClient(lambda: rebuild()._async_client)
    return model


@functools.cache
def _per_loop_httpx_client_class() -> type:
    import httpx

    class PerLoopHttpxAsyncClient(httpx.AsyncClient):
        """An ``httpx.AsyncClient`` that sends through the loop's own client.

        The OpenAI client takes an ``httpx.AsyncClient`` (``http_async_client``)
        and sends every request through its ``send``; this one hands each
        request to the running loop's client (``PerLoopAsyncClient``).
        """

        def __init__(self, clients: PerLoopAsyncClient, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._per_loop = clients

        async def send(self, request: httpx.Request, **kwargs: Any) -> Any:
            return await self._per_loop.send(request, **kwargs)

    return PerLoopHttpxAsyncClient


def openai_http_async_client(timeout: float, transport: Any = None) -> Any:
    """``http_async_client`` for ``ChatOpenAI`` / ``OpenAIEmbeddings``.

    Without one, a chat model uses langchain-openai's process-wide async
    client and an embedding model one of its own, kept across event loops:
    the first call on every later loop meets a keep-alive connection of the
    closed one, fails with "Event loop is closed" and the OpenAI client sends
    it again (billed twice, and failed with ``max_attempts: 1``). This client
    sends through one client per event loop, without keep-alive, as for
    Ollama.

    ``transport`` (tests: ``httpx.MockTransport``) opens no connection, so a
    plain client on it is returned.
    """
    httpx = importlib.import_module("httpx")
    if transport is not None:
        return httpx.AsyncClient(transport=transport, timeout=timeout)

    def build() -> Any:
        # The OpenAI client's own defaults, but for keep-alive.
        return httpx.AsyncClient(
            timeout=timeout,
            limits=_no_keepalive_limits(),
            follow_redirects=True,
            verify=tls_context(),
        )

    return _per_loop_httpx_client_class()(
        PerLoopAsyncClient(build),
        timeout=timeout,
        follow_redirects=True,
        verify=tls_context(),
    )


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
    """``callbacks`` plus the usage-accounting handler, and the GenAI tracing
    handler when tracing is enabled."""
    from unified_kg_rag.adapters.llm.langchain import with_callback
    from unified_kg_rag.adapters.observability.usage_callback import (
        UsageCallbackHandler,
    )

    callbacks = with_callback(
        callbacks, UsageCallbackHandler(model_id, role=purpose.value)
    )
    if not observability.is_enabled():
        return callbacks
    from unified_kg_rag.adapters.observability.langchain_callback import (
        GenAITracingCallbackHandler,
    )

    handler: BaseCallbackHandler = GenAITracingCallbackHandler(
        model_id,
        provider_name=provider_name,
        purpose=purpose.value,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return with_callback(callbacks, handler)


def local_transient_retry(config: Config) -> TransientRetryConfig:
    """Retry policy for a client that does not retry on its own (Ollama).

    ``retry.model_server_max_attempts`` attempts with the backoff and
    wall-clock budget of the ``retry`` section.
    """
    retry = config.retry
    return retry.transient_policy(max_attempts=retry.model_server_max_attempts)


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

    @staticmethod
    def is_item_error(exc: BaseException) -> bool:
        """Whether an ingestion item's error counts against its document:
        HTTP 400/413/422 other than a rejected model or setting
        (:func:`~unified_kg_rag.adapters.llm.retry.looks_like_item_error`)."""
        return looks_like_item_error(exc)

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
                    self.config.models.default_max_output_tokens,
                    kwargs.get("min_output_tokens") or 0,
                )
            ),
            info.max_output_tokens,
        )
        # A short-answer query prompt caps it (search.output_ceilings), but
        # not a model that may reason: these servers take no thinking budget
        # to leave room for, so a ceiling would cut it off in its reasoning
        # (qwen3, deepseek-r1), and the model keeps its whole cap.
        if (
            not kwargs.get("max_tokens")
            and (ceiling := kwargs.get("output_ceiling"))
            and not self.may_reason(model_id)
        ):
            max_tokens = ceiled_output_tokens(max_tokens, int(ceiling), 0)
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
        model = self._build(model_id, max_tokens, temperature, callbacks, purpose)
        # nosemgrep: python.lang.security.audit.logging.logger-credential-leak.python-logger-credential-disclosure - LLM token counts, not credentials
        logger.debug(
            "Created %s language model '%s' (max_tokens %s)",
            self.kind,
            model_id,
            max_tokens,
        )
        return model

    def may_reason(self, model_id: str) -> bool:
        """Whether ``model_id`` may reason before it answers: an
        OpenAI-compatible server does not say, so it may."""
        return True

    def _build(
        self,
        model_id: str,
        max_tokens: int,
        temperature: float,
        callbacks: Any,
        purpose: ModelPurpose,
    ) -> BaseChatModel:
        raise NotImplementedError


# Where OpenAI-compatible servers that reason (vLLM's and SGLang's reasoning
# parsers, DeepSeek, llama.cpp) return the reasoning, beside the answer.
_REASONING_FIELDS = ("reasoning_content", "reasoning")


def _reasoning_of(choice: Any) -> str | None:
    """The reasoning a chat completion choice returned beside its answer."""
    message = choice.get("message") if isinstance(choice, dict) else choice.message
    if message is None:
        return None
    fields = (
        message
        if isinstance(message, dict)
        else {**(getattr(message, "model_extra", None) or {}), **vars(message)}
    )
    for name in _REASONING_FIELDS:
        if isinstance(value := fields.get(name), str) and value:
            return value
    return None


@functools.cache
def _reasoning_chat_openai_class(module: Any) -> type:
    """``ChatOpenAI`` that keeps a server's separate reasoning.

    ``ChatOpenAI`` reads the official API's fields only and drops a
    ``reasoning_content`` (or ``reasoning``) field; this one puts it in the
    message's ``additional_kwargs["reasoning_content"]``, where
    ``ChatOllama`` puts it, so the chain normaliser (``ReasoningStripper``)
    reads one place for both servers.
    """

    class ReasoningChatOpenAI(module.ChatOpenAI):
        def _create_chat_result(
            self, response: Any, generation_info: Any = None
        ) -> Any:
            result = super()._create_chat_result(response, generation_info)
            choices = (
                response.get("choices")
                if isinstance(response, dict)
                else getattr(response, "choices", None)
            ) or []
            for generation, choice in zip(result.generations, choices, strict=False):
                if reasoning := _reasoning_of(choice):
                    generation.message.additional_kwargs["reasoning_content"] = (
                        reasoning
                    )
            return result

    return ReasoningChatOpenAI


class OpenAICompatibleLanguageModelFactory(_LocalLanguageModelFactory):
    """Chat models on an OpenAI-compatible server (``ChatOpenAI``).

    ``providers.llm.base_url`` points at the server's ``/v1`` root (vLLM,
    llama.cpp server, LM Studio, or OpenAI itself when unset).

    The OpenAI client retries transient errors itself
    (``retry.model_server_max_attempts``), so this factory defines no
    ``transient_retry`` and query chains are not wrapped in a second retry.
    """

    provider_name = "openai"
    kind = "openai_compatible"

    def _build(
        self,
        model_id: str,
        max_tokens: int,
        temperature: float,
        callbacks: Any,
        purpose: ModelPurpose,
    ) -> BaseChatModel:
        module = import_optional("langchain_openai", self.kind)
        timeout = request_timeout(self.config, self.settings.timeout_seconds, purpose)
        extra: dict[str, Any] = {
            "http_async_client": openai_http_async_client(timeout, self.transport)
        }
        if self.transport is not None:
            httpx = importlib.import_module("httpx")
            extra["http_client"] = httpx.Client(
                transport=self.transport, timeout=timeout
            )
        model: BaseChatModel = _reasoning_chat_openai_class(module)(
            model=model_id,
            base_url=self.settings.base_url,
            api_key=api_key_value(self.settings),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            # The OpenAI client counts retries after the first attempt.
            max_retries=self.config.retry.model_server_max_attempts - 1,
            callbacks=callbacks,
            **extra,
        )
        return model


# Seconds the capability lookup (``/api/show``) of a model may take.
_SHOW_TIMEOUT_SECONDS = 10.0


class OllamaLanguageModelFactory(_LocalLanguageModelFactory):
    """Chat models on an Ollama server (``ChatOllama``).

    ``providers.llm.context_window_size`` is sent as ``num_ctx``: Ollama's own
    default window is far smaller than the prompts the pipeline sends, and it
    truncates the prompt silently when it does not fit.

    The Ollama client never retries, so ``transient_retry`` (from
    ``retry.model_server_max_attempts``) gives query chains the chain-level retry
    of ``setup_chain``; ingestion chains keep the batch retries.
    """

    provider_name = "ollama"
    kind = "ollama"

    def __init__(self, config: Config, *, transport: Any = None) -> None:
        super().__init__(config, transport=transport)
        self._thinks: dict[str, bool] = {}
        self._thinks_lock = threading.Lock()

    @property
    def transient_retry(self) -> TransientRetryConfig:
        """The retry policy ``setup_chain`` applies to query chains."""
        return local_transient_retry(self.config)

    def may_reason(self, model_id: str) -> bool:
        """Whether the server lists ``thinking`` among ``model_id``'s
        capabilities (:meth:`thinks`)."""
        return self.thinks(model_id)

    def thinks(self, model_id: str) -> bool:
        """Whether the server lists ``thinking`` among ``model_id``'s
        capabilities (``/api/show``), asked once per model.

        Such a model is built with ``reasoning=True``: its reasoning comes
        back beside the answer (``additional_kwargs["reasoning_content"]``)
        instead of as a ``<think>`` block in it. Ollama refuses that setting
        for a model without the capability (HTTP 400 "does not support
        thinking"), and an older server lists no capabilities, so anything
        but a listed ``thinking`` keeps the server's default; the chain
        normaliser strips a block left in the answer either way.
        """
        with self._thinks_lock:
            if (known := self._thinks.get(model_id)) is not None:
                return known
        module = import_optional("ollama", self.kind)
        # Short: building a model made no request before, and a server that
        # does not answer this fails the model's first call anyway.
        kwargs: dict[str, Any] = {
            "timeout": min(self.settings.timeout_seconds, _SHOW_TIMEOUT_SECONDS),
            "verify": tls_context(),
        }
        if self.transport is not None:
            kwargs["transport"] = self.transport
        try:
            shown = module.Client(host=self.settings.base_url, **kwargs).show(model_id)
            capabilities = getattr(shown, "capabilities", None) or []
        except Exception as e:  # noqa: BLE001 - the server's default is kept
            logger.debug("Ollama capabilities of '%s' unknown: %s", model_id, e)
            capabilities = []
        thinks = "thinking" in capabilities
        with self._thinks_lock:
            self._thinks[model_id] = thinks
        return thinks

    def _build(
        self,
        model_id: str,
        max_tokens: int,
        temperature: float,
        callbacks: Any,
        purpose: ModelPurpose,
    ) -> BaseChatModel:
        module = import_optional("langchain_ollama", self.kind)
        timeout = request_timeout(self.config, self.settings.timeout_seconds, purpose)
        client_kwargs = ollama_client_kwargs(timeout)
        # A query call's async client gets the timeout too (the async one
        # otherwise has none); the other purposes keep theirs as they were.
        async_timeout = timeout if purpose is ModelPurpose.QUERY else None
        if self.transport is not None:
            client_kwargs["transport"] = self.transport
        reasoning = {"reasoning": True} if self.thinks(model_id) else {}

        def build() -> BaseChatModel:
            chat: BaseChatModel = module.ChatOllama(
                model=model_id,
                base_url=self.settings.base_url,
                temperature=temperature,
                num_predict=max_tokens,
                num_ctx=self.settings.context_window_size,
                client_kwargs=client_kwargs,
                async_client_kwargs=ollama_async_client_kwargs(async_timeout),
                callbacks=callbacks,
                **reasoning,
            )
            return chat

        model: BaseChatModel = per_loop_async_client(build(), build)
        return model
