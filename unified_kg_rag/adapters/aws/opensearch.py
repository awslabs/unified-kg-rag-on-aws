# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

import boto3
from opensearchpy import (
    AsyncHttpConnection,
    AsyncOpenSearch,
    AWSV4SignerAsyncAuth,
    AWSV4SignerAuth,
    OpenSearch,
    RequestsHttpConnection,
)
from opensearchpy.exceptions import NotFoundError, TransportError
from opensearchpy.helpers import streaming_bulk

from unified_kg_rag.adapters.aws.bedrock_retry import backoff_delay
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import AWSServiceError, get_logger

logger = get_logger(__name__)

# Module-level indirection so tests can stub the bulk-item backoff wait.
_sleep = time.sleep

# Per-item statuses inside a bulk response worth resending: the cluster
# rejected the item under load (429, e.g. a full write queue) or a node was
# briefly unavailable. The client's retry_on_status only covers the status of
# the whole request, and a bulk request with rejected items still returns 200.
RETRYABLE_BULK_ITEM_STATUSES = frozenset({429, 502, 503, 504})


def _bulk_item_status(result: dict[str, Any]) -> int | None:
    item: Any = next(iter(result.values()), {})
    status = item.get("status") if isinstance(item, dict) else None
    return status if isinstance(status, int) else None


def _handle_opensearch_errors(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(client_instance: "OpenSearchClient", *args: Any, **kwargs: Any) -> Any:
        try:
            return func(client_instance, *args, **kwargs)
        except NotFoundError as e:
            logger.debug("Resource not found during '%s': %s", func.__name__, e)
            raise
        except TransportError as e:
            msg = f"Transport error in '{func.__name__}': status={e.status_code}, info={e.info}"
            logger.exception(msg)
            raise AWSServiceError(msg) from e
        except Exception as e:
            msg = f"Unexpected error in '{func.__name__}': {e}"
            logger.exception(msg)
            raise AWSServiceError(msg) from e

    return wrapper


def _handle_async_opensearch_errors(func: Callable) -> Callable:
    @wraps(func)
    async def async_wrapper(
        client_instance: "OpenSearchClient", *args: Any, **kwargs: Any
    ) -> Any:
        try:
            return await func(client_instance, *args, **kwargs)
        except NotFoundError as e:
            logger.debug("Resource not found during async '%s': %s", func.__name__, e)
            raise
        except TransportError as e:
            msg = f"Async transport error in '{func.__name__}': status={e.status_code}, info={e.info}"
            logger.exception(msg)
            raise AWSServiceError(msg) from e
        except Exception as e:
            msg = f"Unexpected async error in '{func.__name__}': {e}"
            logger.exception(msg)
            raise AWSServiceError(msg) from e

    return async_wrapper


# Strong references to close tasks scheduled from sync code, so they are not
# garbage-collected before they finish.
_PENDING_CLOSES: set[asyncio.Task[Any]] = set()


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


def _finish_without_loop(awaitable: Any) -> None:
    """Run an awaitable whose event loop is gone, as far as it can go.

    aiohttp's ``TCPConnector.close()`` is a coroutine: un-awaited it closes
    nothing (and warns "never awaited"). Once the connector's loop is closed
    its close steps complete without suspending, so stepping the coroutine
    directly finishes it. If it does suspend, nothing could resume it, so it
    is closed instead. A Future/Task belongs to a loop and is left alone.
    """
    if isinstance(awaitable, asyncio.Future) or not hasattr(awaitable, "__await__"):
        return
    iterator = awaitable.__await__()
    try:
        iterator.send(None)
    except StopIteration:
        return
    close = getattr(iterator, "close", None)
    if close is not None:
        close()


class OpenSearchClient:
    # Resends of bulk items rejected with a retryable status (see
    # RETRYABLE_BULK_ITEM_STATUSES), with exponential backoff between rounds.
    BULK_ITEM_MAX_RETRIES = 4
    BULK_ITEM_BASE_DELAY_SECONDS = 1.0
    BULK_ITEM_MAX_DELAY_SECONDS = 30.0
    BULK_CHUNK_SIZE = 100

    def __init__(self, config: Config, boto_session: boto3.Session | None = None):
        self.config = config
        self.opensearch_config = config.aws.opensearch
        self.boto_session = boto_session or boto3.Session(
            profile_name=config.aws.profile_name,
            region_name=config.aws.region_name,
        )
        self._client: OpenSearch | None = None
        self._async_client: AsyncOpenSearch | None = None
        self._bound_loop_id: int | None = None
        # The loop object itself: keeps its id from being reused while bound,
        # and lets a rotated client be closed (awaited) on its own loop.
        self._bound_loop: asyncio.AbstractEventLoop | None = None

    @property
    def client(self) -> OpenSearch:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def async_client(self) -> AsyncOpenSearch:
        current_loop_id = self._get_current_loop_id()

        if self._async_client is None or (
            current_loop_id is not None and self._bound_loop_id != current_loop_id
        ):
            if self._async_client is not None:
                logger.debug(
                    "Event loop changed (old=%s, new=%s), recreating async client",
                    self._bound_loop_id,
                    current_loop_id,
                )
                # The previous AsyncOpenSearch is bound to a now-defunct event
                # loop; abandon it without awaiting (we are in a sync property
                # and cannot await its aclose() here) so the replacement does
                # not silently leak the old aiohttp session/connection pool.
                # Best effort: close the underlying connector(s) synchronously.
                self._discard_async_client(
                    self._async_client, getattr(self, "_bound_loop", None)
                )
            self._async_client = self._create_async_client()
            self._bound_loop_id = current_loop_id
            self._bound_loop = _running_loop()

        return self._async_client

    @staticmethod
    def _discard_async_client(
        async_client: AsyncOpenSearch,
        bound_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        """Best-effort close of an async client from sync code. Never raises.

        Used when the bound event loop has rotated, or from ``close()``. While
        the client's loop is still running, its awaited ``close()`` is
        scheduled there (the clean path). Otherwise the loop is gone, and the
        underlying aiohttp connectors' ``close()`` coroutines are run to
        completion without a loop (see ``_finish_without_loop``).
        """
        try:
            if (
                bound_loop is not None
                and bound_loop.is_running()
                and not bound_loop.is_closed()
            ):
                if bound_loop is _running_loop():
                    task = bound_loop.create_task(async_client.close())
                    _PENDING_CLOSES.add(task)
                    task.add_done_callback(_PENDING_CLOSES.discard)
                else:
                    asyncio.run_coroutine_threadsafe(async_client.close(), bound_loop)
                return
            transport = getattr(async_client, "transport", None)
            pool = getattr(transport, "connection_pool", None)
            for connection in getattr(pool, "connections", []) or []:
                session = getattr(connection, "session", None)
                connector = getattr(session, "connector", None)
                if connector is not None:
                    _finish_without_loop(connector.close())
        except Exception as e:  # noqa: BLE001 - teardown must never raise
            logger.debug("Could not eagerly close rotated async client: %s", e)

    def _get_current_loop_id(self) -> int | None:
        try:
            return id(asyncio.get_running_loop())
        except RuntimeError:
            return None

    def close(self) -> None:
        """Close the sync HTTP client and best-effort the async one.

        Mirrors :class:`NeptuneClient.close`: idempotent and never raises. The
        async client's transport exposes a coroutine ``close()``; from sync code
        we can only eagerly close its connector (see ``aclose`` for the awaited
        path). Releases the requests connection pool so a process that finishes
        a query/ingest does not leak sockets.
        """
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:  # noqa: BLE001 - teardown must never raise
                logger.debug("Error closing sync OpenSearch client: %s", e)
            self._client = None
        if self._async_client is not None:
            self._discard_async_client(
                self._async_client, getattr(self, "_bound_loop", None)
            )
            self._async_client = None
            self._bound_loop_id = None
            self._bound_loop = None

    async def aclose(self) -> None:
        """Async teardown: await the AsyncOpenSearch transport close.

        This is the correct path when running inside an event loop — it awaits
        ``AsyncOpenSearch.close()`` so the aiohttp session is closed cleanly
        (avoiding the "Unclosed client session" warning) rather than only
        dropping the connector. The sync client is closed too. Never raises.
        """
        if self._async_client is not None:
            try:
                await self._async_client.close()
            except Exception as e:  # noqa: BLE001 - teardown must never raise
                logger.debug("Error closing async OpenSearch client: %s", e)
            self._async_client = None
            self._bound_loop_id = None
            self._bound_loop = None
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:  # noqa: BLE001 - teardown must never raise
                logger.debug("Error closing sync OpenSearch client: %s", e)
            self._client = None

    def __enter__(self) -> "OpenSearchClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    async def __aenter__(self) -> "OpenSearchClient":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.aclose()

    def _create_client(self) -> OpenSearch:
        params = self._get_base_connection_params(async_mode=False)
        params.update(
            {
                "connection_class": RequestsHttpConnection,
                "max_retries": 5,
                "retry_on_timeout": True,
                "retry_on_status": (429, 502, 503, 504),
            }
        )

        try:
            client = OpenSearch(**params)
            if not client.ping():
                raise ConnectionError("OpenSearch cluster is not reachable.")

            info = client.info()
            cluster_name = info.get("cluster_name", "unknown")
            logger.info("Connected to OpenSearch cluster: %s", cluster_name)
            return client
        except Exception as e:
            logger.exception("Failed to create OpenSearch client: %s", e)
            raise AWSServiceError("Failed to connect to OpenSearch.") from e

    def _create_async_client(self) -> AsyncOpenSearch:
        params = self._get_base_connection_params(async_mode=True)
        # AsyncHttpConnection (NOT AIOHttpConnection): the latter unconditionally
        # treats http_auth as a basic-auth credential and calls .encode() on it,
        # so an AWSV4SignerAsyncAuth still breaks. AsyncHttpConnection invokes a
        # callable http_auth as a per-request SigV4 signer instead.
        params["connection_class"] = AsyncHttpConnection
        # Match the sync client's retry policy: the async client serves the whole
        # query hot path (every retriever fans several index queries out
        # concurrently). Without this, opensearch-py defaults to
        # retry_on_timeout=False and retry_on_status=(502,503,504) — a 429
        # (throttling) would surface as a TransportError and silently degrade
        # that index to zero results instead of being retried.
        params.update(
            {
                "max_retries": 5,
                "retry_on_timeout": True,
                "retry_on_status": (429, 502, 503, 504),
            }
        )
        return AsyncOpenSearch(**params)

    def _get_base_connection_params(self, async_mode: bool) -> dict[str, Any]:
        if not self.opensearch_config.endpoint:
            raise AWSServiceError("OpenSearch endpoint is not configured.")

        return {
            "hosts": [
                {
                    "host": self.opensearch_config.endpoint,
                    "port": self.opensearch_config.port,
                }
            ],
            "http_auth": self._get_auth(async_mode=async_mode),
            "use_ssl": self.opensearch_config.use_ssl,
            "verify_certs": self.opensearch_config.verify_certs,
            "timeout": 180,
        }

    def _get_auth(
        self, async_mode: bool
    ) -> AWSV4SignerAuth | AWSV4SignerAsyncAuth | tuple[str, str] | None:
        if self.opensearch_config.use_iam:
            creds = self.boto_session.get_credentials()
            if not creds:
                raise AWSServiceError("Cannot get AWS credentials for OpenSearch IAM.")
            # opensearch-py's own SigV4 signers must be used, NOT
            # requests_aws4auth.AWS4Auth: the sync RequestsHttpConnection tolerates
            # AWS4Auth, but AIOHttpConnection treats http_auth as a basic-auth
            # credential and calls .encode() on it -> "'AWS4Auth' object has no
            # attribute 'encode'", silently returning zero search hits. The async
            # connection needs AWSV4SignerAsyncAuth; the sync one AWSV4SignerAuth.
            signer = AWSV4SignerAsyncAuth if async_mode else AWSV4SignerAuth
            # 'es' (managed domain) by default; 'aoss' for OpenSearch Serverless.
            return signer(
                creds,
                self.config.aws.region_name,
                self.opensearch_config.sigv4_service_name,
            )

        if self.opensearch_config.username and self.opensearch_config.password:
            return (
                self.opensearch_config.username,
                self.opensearch_config.password.get_secret_value(),
            )

        if self.opensearch_config.allow_anonymous:
            logger.debug("Connecting to OpenSearch without authentication")
            return None

        raise AWSServiceError(
            "No OpenSearch auth method configured (IAM, basic or allow_anonymous)."
        )

    @_handle_opensearch_errors
    def bulk_index(
        self, index: str, documents: list[dict[str, Any]], refresh: bool = True
    ) -> dict[str, Any]:
        if not documents:
            return {"errors": False, "items": []}

        actions: list[dict[str, Any]] = []
        for doc in documents:
            action = {"_op_type": "index", "_index": index, "_source": doc}
            if (doc_id := doc.get("id")) is not None:
                action["_id"] = str(doc_id)
            actions.append(action)

        try:
            # Rejected documents are collected into `errors` and reported
            # rather than aborting the batch mid-stream (matches bulk_delete).
            success_count, errors = self._bulk_with_item_retries(
                actions, lambda ok, _result: ok
            )
        except Exception as e:
            raise AWSServiceError("Streaming bulk operation failed.") from e

        if errors:
            logger.warning(
                "Bulk index completed with %s errors in '%s'", len(errors), index
            )
        else:
            logger.debug(
                "Successfully indexed %s documents to '%s'", success_count, index
            )

        if refresh and success_count > 0:
            self.client.indices.refresh(index=index)

        return {"errors": bool(errors), "items": errors}

    def bulk_delete(
        self, index: str, ids: list[str], refresh: bool = True
    ) -> dict[str, Any]:
        """Delete documents by id from ``index`` (used by incremental removals).

        Missing ids are ignored. Returns the same ``{errors, items}`` shape as
        :meth:`bulk_index`.
        """
        if not ids:
            return {"errors": False, "items": []}

        actions = [
            {"_op_type": "delete", "_index": index, "_id": str(doc_id)}
            for doc_id in ids
        ]

        def deleted(ok: bool, result: dict[str, Any]) -> bool:
            # A delete of a missing doc reports not_found; treat as ok.
            return ok or result.get("delete", {}).get("result") == "not_found"

        try:
            success_count, errors = self._bulk_with_item_retries(actions, deleted)
        except Exception as e:
            raise AWSServiceError("Streaming bulk delete failed.") from e

        if refresh and success_count > 0:
            self.client.indices.refresh(index=index)

        return {"errors": bool(errors), "items": errors}

    def _bulk_with_item_retries(
        self,
        actions: list[dict[str, Any]],
        succeeded: Callable[[bool, dict[str, Any]], bool],
    ) -> tuple[int, list[dict[str, Any]]]:
        """Send ``actions`` in bulk, resending items rejected with a retryable status.

        ``streaming_bulk`` (``raise_on_error=False``) yields one result per
        action, in order, so each failed result maps back to its action. Items
        that failed with a status in :data:`RETRYABLE_BULK_ITEM_STATUSES` are
        resent up to :attr:`BULK_ITEM_MAX_RETRIES` times with backoff; other
        failures are final. Returns the success count and the failed results
        (each ``{op_type: {"_id", "status", "error", ...}}``).
        """
        success_count = 0
        failures: list[dict[str, Any]] = []
        retry_results: list[dict[str, Any]] = []
        pending = actions
        for attempt in range(self.BULK_ITEM_MAX_RETRIES + 1):
            if attempt:
                delay = backoff_delay(
                    attempt,
                    self.BULK_ITEM_BASE_DELAY_SECONDS,
                    self.BULK_ITEM_MAX_DELAY_SECONDS,
                )
                logger.warning(
                    "Resending %s bulk item(s) rejected with a retryable status "
                    "(retry %s/%s) in %.1fs",
                    len(pending),
                    attempt,
                    self.BULK_ITEM_MAX_RETRIES,
                    delay,
                )
                _sleep(delay)
            retry_actions: list[dict[str, Any]] = []
            retry_results = []
            results = streaming_bulk(
                client=self.client,
                actions=iter(pending),
                chunk_size=self.BULK_CHUNK_SIZE,
                raise_on_error=False,
            )
            for action, (ok, result) in zip(pending, results, strict=True):
                if succeeded(ok, result):
                    success_count += 1
                elif _bulk_item_status(result) in RETRYABLE_BULK_ITEM_STATUSES:
                    retry_actions.append(action)
                    retry_results.append(result)
                else:
                    failures.append(result)
            if not retry_actions:
                return success_count, failures
            pending = retry_actions
        logger.warning(
            "%s bulk item(s) still rejected after %s retries",
            len(retry_results),
            self.BULK_ITEM_MAX_RETRIES,
        )
        return success_count, failures + retry_results

    @_handle_opensearch_errors
    def create_index(self, index: str, body: dict[str, Any]) -> None:
        if not self.client.indices.exists(index=index):
            self.client.indices.create(index=index, body=body)
            logger.info("Created index: '%s'", index)

    @_handle_opensearch_errors
    def create_search_pipeline(self, pipeline_id: str, body: dict[str, Any]) -> None:
        if not self.check_search_pipeline_exists(pipeline_id):
            self.client.search_pipeline.put(id=pipeline_id, body=body)
            logger.debug("Created search pipeline: '%s'", pipeline_id)

    @_handle_opensearch_errors
    def check_search_pipeline_exists(self, pipeline_id: str) -> bool:
        try:
            self.client.search_pipeline.get(id=pipeline_id)
            return True
        except NotFoundError:
            return False

    @_handle_opensearch_errors
    def delete_alias(
        self, index_names: str | list[str], alias_names: str | list[str]
    ) -> None:
        final_aliases = (
            ",".join(alias_names) if isinstance(alias_names, list) else alias_names
        )
        final_indices = (
            ",".join(index_names) if isinstance(index_names, list) else index_names
        )

        logger.info(
            "Deleting alias(es) '%s' from index(es) '%s'", final_aliases, final_indices
        )
        self.client.indices.delete_alias(index=final_indices, name=final_aliases)

    @_handle_opensearch_errors
    def delete_indices(self, index_names: list[str]) -> None:
        if index_names:
            indices_str = ",".join(index_names)
            self.client.indices.delete(index=indices_str)
            logger.info("Deleted indices: %s", indices_str)

    @_handle_opensearch_errors
    def get_index_name_by_alias(self, alias_name: str) -> str | None:
        indices = self.get_indices_by_alias(alias_name)
        return indices[0] if indices else None

    @_handle_opensearch_errors
    def get_indices_by_alias(self, alias_name: str) -> list[str]:
        try:
            return list(self.client.indices.get_alias(name=alias_name).keys())
        except NotFoundError:
            return []

    @_handle_opensearch_errors
    def get_aliases_by_index(self, index_pattern: str) -> dict[str, list[str]]:
        """Map every index whose *name* matches ``index_pattern`` to its aliases.

        Unlike ``get_indices_by_alias`` (which matches alias names), this
        matches index names, so it also returns indices that no alias points
        at any more — the stale ones a blue/green swap leaves behind.
        """
        try:
            response = self.client.indices.get_alias(
                index=index_pattern, expand_wildcards="open,closed"
            )
        except NotFoundError:
            return {}
        return {
            name: sorted((body or {}).get("aliases") or {})
            for name, body in response.items()
        }

    @_handle_opensearch_errors
    def search(self, **kwargs: Any) -> dict[str, Any]:
        result = self.client.search(**kwargs)
        return dict(result)

    @_handle_async_opensearch_errors
    async def aindex_exists(self, index: str) -> bool:
        """Whether an index or alias named ``index`` exists."""
        return bool(await self.async_client.indices.exists(index=index))

    @_handle_async_opensearch_errors
    async def asearch(self, **kwargs: Any) -> dict[str, Any]:
        result = await self.async_client.search(**kwargs)
        return dict(result)

    @_handle_opensearch_errors
    def update_alias(
        self,
        alias_name: str,
        new_index_name: str,
        remove_pattern: str | None = None,
    ) -> None:
        actions = []

        if remove_pattern:
            actions.append(
                {
                    "remove": {
                        "index": remove_pattern,
                        "alias": alias_name,
                        "must_exist": False,
                    }
                }
            )

        actions.append({"add": {"index": new_index_name, "alias": alias_name}})

        body = {"actions": actions}
        self.client.indices.update_aliases(body=body)
        logger.info("Updated alias '%s' to point to '%s'", alias_name, new_index_name)
