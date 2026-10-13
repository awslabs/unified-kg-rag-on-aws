# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import copy
import threading
from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING, Any

import aiohttp
from gremlin_python.driver.aiohttp.transport import AiohttpTransport
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.driver.protocol import GremlinServerError
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import GraphTraversalSource
from gremlin_python.process.traversal import Traversal

from unified_kg_rag.adapters.store_errors import causes
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import (
    AWSServiceError,
    GraphRAGException,
    StoreConnectionError,
    get_logger,
)
from unified_kg_rag.shared.exceptions import (
    StoreAccessDeniedError,
    StoreConfigurationError,
    StoreTimeoutError,
)

if TYPE_CHECKING:
    import boto3

logger = get_logger(__name__)

# Neptune engine error codes for requests that fail the same way on every try
# (Neptune user guide, "Graph Engine Error Messages and Codes"). Retrying
# them only multiplies the failure, so they fail fast; any other error keeps
# being retried. A connection the server closed is replaced first
# (``run_traversal``), so the retry reaches the server again.
PERMANENT_NEPTUNE_ERROR_CODES: frozenset[str] = frozenset(
    {
        "AccessDeniedException",
        "BadRequestException",
        "InvalidNumericDataException",
        "InvalidParameterException",
        "MalformedQueryException",
        "MethodNotAllowedException",
        "MissingParameterException",
        "ReadOnlyViolationException",
        "UnsupportedOperationException",
    }
)


def is_permanent_neptune_error(exc: BaseException) -> bool:
    """Return True if a Gremlin request failed in a way a retry cannot fix.

    Neptune reports engine errors as a ``GremlinServerError`` whose status
    message carries the JSON error body, ``"code"`` included; the code names
    are distinct identifiers, so matching them in the message is matching the
    code. Client-side argument errors (``ValueError``/``TypeError``) are
    permanent too, as is a ``ConfigurationError`` (a denied connection).
    """
    if isinstance(exc, GremlinServerError):
        message = str(exc.status_message)
        return any(code in message for code in PERMANENT_NEPTUNE_ERROR_CODES)
    return isinstance(exc, ValueError | TypeError)


# What gremlinpython raises on a websocket that is gone (the server closed it,
# or restarted, or failed over): the pooled connection is never reopened.
_CLOSED_CONNECTION_MESSAGES = (
    "Connection was already closed.",
    "Connection was closed by server.",
)


def is_connection_lost(exc: BaseException) -> bool:
    """Whether ``exc`` (or a cause) means the Gremlin connection is unusable.

    The websocket could not be opened (refused, DNS, TLS: aiohttp's
    ``ClientConnectorError``), was reset, or was closed by the server. Every
    later request on that connection fails the same way until it is replaced.
    A timeout is not one (see ``is_timeout``).
    """
    if is_timeout(exc):
        # aiohttp's ServerTimeoutError is a ClientConnectionError too, but a
        # server too slow to answer may answer the next request.
        return False
    for current in causes(exc):
        if isinstance(current, ConnectionError | aiohttp.ClientConnectionError):
            return True
        if isinstance(current, RuntimeError) and str(current) in (
            _CLOSED_CONNECTION_MESSAGES
        ):
            return True
    return False


def is_timeout(exc: BaseException) -> bool:
    """Whether ``exc`` (or a cause) is a request that got no answer in time."""
    # asyncio.TimeoutError is not the builtin one before Python 3.11.
    return any(
        isinstance(current, TimeoutError | asyncio.TimeoutError)
        for current in causes(exc)
    )


def neptune_request_error(
    exc: Exception, timeout_seconds: float | None, url: str | None = None
) -> Exception:
    """The error a failed Gremlin request raises: typed when Neptune is unusable.

    A lost or unopenable connection becomes a ``StoreConnectionError``, a
    request without an answer within ``timeout_seconds`` a
    ``StoreTimeoutError`` and an IAM denial a ``StoreAccessDeniedError``; this
    package's own errors and any other failure (a query Neptune failed) are
    returned unchanged.
    """
    if isinstance(exc, GraphRAGException):
        return exc
    if isinstance(exc, GremlinServerError) and "AccessDeniedException" in str(
        exc.status_message
    ):
        return StoreAccessDeniedError(
            f"Neptune denied the request: {exc}. Check that the caller's IAM "
            "policy allows the neptune-db actions the query needs (such as "
            "neptune-db:ReadDataViaQuery)."
        )
    if is_timeout(exc):
        within = f"within {timeout_seconds} s " if timeout_seconds else ""
        return StoreTimeoutError(
            f"Neptune did not answer {within}(search.store_timeout_seconds)"
        )
    if is_connection_lost(exc):
        at = f" at '{url}'" if url else ""
        return StoreConnectionError(
            f"Neptune is not reachable{at} (Gremlin connection failed or lost): {exc}"
        )
    return exc


def run_traversal(
    client: "NeptuneClient",
    build: Callable[[GraphTraversalSource], Traversal],
    *,
    results: bool = False,
) -> list[Any]:
    """Run the traversal ``build`` makes from ``client.g``; with ``results``, return them.

    The traversal is built per call, from the current source, so a caller that
    retries reaches the server over a new connection: when Neptune is
    unusable (a connection the server closed on a restart or failover, or a
    request without an answer) the connection is dropped (``reset``) and the
    error raised as a ``StoreConnectionError``. gremlinpython never reopens a
    pooled websocket the server closed, so reusing it failed every retry.
    Without ``results`` the traversal is iterated (nothing sent back) and
    ``[]`` is returned.
    """
    g = client.g
    try:
        traversal = build(g)
        if results:
            return list(traversal.toList())
        traversal.iterate()
        return []
    except Exception as e:
        error = neptune_request_error(e, client.timeout_seconds)
        if isinstance(error, StoreConnectionError):
            client.reset(g)
        if error is e:
            raise
        raise error from e


class _NeptuneTransport(AiohttpTransport):
    """gremlinpython's websocket transport, bounded in time and signed per socket.

    gremlinpython opens each pooled websocket on its first request, in an
    aiohttp session with its 5-minute default timeout and with the headers of
    the moment the ``DriverRemoteConnection`` was built. A SigV4 signature is
    valid for five minutes, so a pooled socket first used later was refused;
    here every handshake is signed anew (``sign``). ``timeout`` bounds the
    handshake and each read and write (``None``: unbounded, gremlinpython's
    default), so an unresponsive server fails the request instead of blocking
    its thread forever.
    """

    def __init__(
        self,
        timeout: float | None,
        sign: Callable[[str], dict[str, str]] | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(read_timeout=timeout, write_timeout=timeout, **kwargs)
        self._timeout = timeout
        self._sign = sign

    def connect(self, url: str, headers: dict[str, str] | None = None) -> None:
        if self._sign is not None:
            headers = {**(headers or {}), **self._sign(url)}
        session_timeout = (
            aiohttp.ClientTimeout(sock_connect=self._timeout, sock_read=self._timeout)
            if self._timeout is not None
            else None
        )

        async def _connect() -> None:
            self._client_session = (
                aiohttp.ClientSession(timeout=session_timeout)
                if session_timeout is not None
                else aiohttp.ClientSession()
            )
            try:
                self._websocket = await self._client_session.ws_connect(
                    url, **self._aiohttp_kwargs, headers=headers
                )
            except aiohttp.ClientResponseError as err:
                if err.status in (401, 403):
                    # Not chained: the response repr carries the signed headers.
                    raise StoreAccessDeniedError(
                        f"Neptune refused the connection (HTTP {err.status}). "
                        "Check that the caller's IAM policy allows "
                        "neptune-db:connect on the cluster and that "
                        "aws.region_name is the cluster's region (IAM auth "
                        "signs for it)."
                    ) from None
                raise

        try:
            self._loop.run_until_complete(_connect())
        except BaseException:
            # gremlinpython closes only a transport that connected; release
            # this one's session and loop now rather than in __del__ (which
            # fails on a thread running a loop).
            self.close()
            raise


def neptune_pool_size(config: Config) -> int:
    """Connections in a NeptuneClient's pool: the most requests in flight."""
    return max(config.aws.neptune.pool_size, config.indexing.neptune.index_concurrency)


class NeptuneClient:
    def __init__(
        self,
        config: Config,
        boto_session: "boto3.Session | None" = None,
        *,
        timeout_seconds: float | None = None,
    ):
        """``timeout_seconds`` bounds opening a connection and each request's
        read and write; ``None`` leaves them unbounded (the indexer's long
        writes and drops)."""
        self.config = config
        self.neptune_config = config.aws.neptune
        if boto_session is not None:
            self.boto_session = boto_session
        self.timeout_seconds = timeout_seconds
        self._connection: DriverRemoteConnection | None = None
        self._g: GraphTraversalSource | None = None
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        # Connection attempts finished, and how the last one failed: callers
        # that waited on the lock during a failed attempt share its failure
        # instead of each opening (and timing out) one in turn.
        self._attempts = 0
        self._attempt_error: Exception | None = None
        logger.debug("Neptune client initialized")

    @cached_property
    def boto_session(self) -> "boto3.Session":
        """The session IAM auth signs with (a default one unless given).

        Created on first use, so a client of a store without IAM auth (a
        local Gremlin Server or OpenSearch) never resolves an AWS profile,
        nor imports the AWS SDK.
        """
        import boto3

        return boto3.Session(
            profile_name=self.config.aws.profile_name,
            region_name=self.config.aws.region_name,
        )

    @property
    def g(self) -> GraphTraversalSource:
        attempts = self._attempts
        with self._lock:
            connection = self._connected(attempts)
            if self._g is None:
                self._g = traversal().withRemote(connection)
            return self._g

    @property
    def connection(self) -> DriverRemoteConnection:
        attempts = self._attempts
        with self._lock:
            return self._connected(attempts)

    def _connected(self, attempts_seen: int) -> DriverRemoteConnection:
        """The open connection, opened now if needed (call holding the lock).

        ``attempts_seen`` is the attempt count the caller read before waiting
        for the lock. If an attempt failed meanwhile, its error is raised
        rather than starting another: concurrent queries of an unreachable
        Neptune wait for one attempt, not one each in turn.
        """
        if self._connection is not None and not self._connection.is_closed():
            return self._connection
        error = self._attempt_error
        if self._attempts != attempts_seen and error is not None:
            # A copy per waiter (same type and message): re-raising the shared
            # error appended every waiter's frames to its traceback and kept
            # them alive with it.
            raise copy.copy(error) from error
        logger.debug("Establishing new Neptune connection")
        try:
            self._connection, self._g = self._create_connection(), None
            self._attempt_error = None
        except Exception as e:
            self._attempt_error = e
            raise
        finally:
            self._attempts += 1
        return self._connection

    def _create_connection(self) -> DriverRemoteConnection:
        """Open a connection and probe it with one request.

        Raises:
            StoreConfigurationError: No endpoint or credentials, or Neptune
                refused the connection (``StoreAccessDeniedError``, HTTP
                401/403).
            StoreConnectionError: Neptune could not be reached
                (``StoreTimeoutError``: it did not answer in time).
            AWSServiceError: The probe failed otherwise (a throttled or failed
                request), which the next use may not.
        """
        if not self.neptune_config.endpoint:
            raise StoreConfigurationError("Neptune endpoint is not configured")

        scheme = "wss" if self.neptune_config.use_ssl else "ws"
        connection_url = (
            f"{scheme}://{self.neptune_config.endpoint}:"
            f"{self.neptune_config.port}/gremlin"
        )
        sign = self._get_auth_headers if self.neptune_config.use_iam else None
        timeout = self.timeout_seconds

        remote_connection: DriverRemoteConnection | None = None
        try:
            # pool_size bounds concurrent in-flight requests over the websocket.
            # Never below indexing.neptune.index_concurrency, so concurrent
            # write batches are multiplexed rather than serialized; max_workers
            # tracks it so result-handling threads are not the bottleneck.
            pool_size = neptune_pool_size(self.config)
            remote_connection = DriverRemoteConnection(
                url=connection_url,
                traversal_source="g",
                pool_size=pool_size,
                max_workers=pool_size,
                transport_factory=lambda: _NeptuneTransport(timeout, sign),
            )
            g = traversal().withRemote(remote_connection)
            g.V().limit(1).toList()
            logger.info(
                "Successfully connected to Neptune at '%s'",
                self.neptune_config.endpoint,
            )
            return remote_connection
        except Exception as e:
            # The probe failed after the websocket and its thread pool opened;
            # nothing else holds the connection, so release it here.
            if remote_connection is not None:
                try:
                    remote_connection.close()
                except Exception as close_error:  # noqa: BLE001 - keep the cause
                    logger.debug("Error closing failed connection: %s", close_error)
            logger.error("Neptune connection probe failed: %s", e)
            error = neptune_request_error(e, timeout, connection_url)
            if error is not e:
                raise error from e
            if isinstance(e, GraphRAGException):
                raise
            raise AWSServiceError(f"Neptune connection probe failed: {e!r}") from e

    def _get_auth_headers(self, url: str) -> dict[str, str]:
        logger.debug("Using IAM authentication for Neptune connection")
        credentials = self.boto_session.get_credentials()
        if not credentials:
            raise StoreConfigurationError(
                "Unable to get AWS credentials for IAM authentication"
            )

        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        try:
            request = AWSRequest(method="GET", url=url, data=None)
            SigV4Auth(
                credentials.get_frozen_credentials(),
                "neptune-db",
                self.config.aws.region_name,
            ).add_auth(request)
            return dict(request.headers.items())
        except Exception as e:
            raise AWSServiceError("Failed to create SigV4 signature for Neptune") from e

    def reset(self, g: GraphTraversalSource | None = None) -> None:
        """Drop the connection, so the next use opens (and signs) a new one.

        gremlinpython never reopens a pooled websocket the server closed (a
        restart or failover): every later request on it fails. With ``g``,
        only while ``g`` is still the current traversal source, so concurrent
        failures on one connection drop it once and never its successor.
        """
        with self._lock:
            if self._connection is None or (g is not None and g is not self._g):
                return
            stale, self._connection, self._g = self._connection, None, None
        try:
            stale.close()
        except Exception as e:  # noqa: BLE001 - the connection is gone anyway
            logger.debug("Error closing a lost Neptune connection: %s", e)
        logger.info("Dropped the Neptune connection; the next query reconnects")

    def close(self) -> None:
        with self._lock:
            if self._connection and not self._connection.is_closed():
                try:
                    self._connection.close()
                    self._connection = None
                    self._g = None
                    logger.info("Closed Neptune connection")
                except Exception as e:
                    logger.error("Error closing Neptune connection: %s", e)

    def __enter__(self) -> "NeptuneClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()
