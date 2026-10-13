# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import functools
import math
import threading
from collections.abc import Callable, Iterator
from functools import cached_property
from typing import TYPE_CHECKING, Any

import aiohttp
from gremlin_python.driver.aiohttp.transport import AiohttpTransport
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.driver.protocol import GremlinServerError
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import GraphTraversalSource, __

from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import (
    AWSServiceError,
    ConfigurationError,
    GraphRAGException,
    StoreConnectionError,
    get_logger,
)

if TYPE_CHECKING:
    import boto3

logger = get_logger(__name__)

# Passes delete_vertices_in_batches allows beyond what the label's count needs.
_EXTRA_DELETE_PASSES = 3

# Neptune engine error codes for requests that fail the same way on every try
# (Neptune user guide, "Graph Engine Error Messages and Codes"). Retrying
# them only multiplies the failure, so they fail fast; any other error,
# including a connection the server closed, keeps being retried.
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
    permanent too.
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


def _causes(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and the exceptions it was raised from, innermost last."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


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
    for current in _causes(exc):
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
        for current in _causes(exc)
    )


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
                    raise ConfigurationError(
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


def _handle_neptune_errors(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self: "NeptuneClient", *args: Any, **kwargs: Any) -> Any:
        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            error_message = f"Neptune operation '{func.__name__}' failed: {e}"
            logger.error(error_message)
            raise AWSServiceError(error_message) from e

    return wrapper


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
        self._timeout_seconds = timeout_seconds
        self._connection: DriverRemoteConnection | None = None
        self._g: GraphTraversalSource | None = None
        self._lock = threading.RLock()  # Reentrant lock for thread safety
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
        with self._lock:
            if self._g is None or self.connection.is_closed():
                self._g = traversal().withRemote(self.connection)
            return self._g

    @property
    def connection(self) -> DriverRemoteConnection:
        with self._lock:
            if self._connection is None or self._connection.is_closed():
                logger.debug("Establishing new Neptune connection")
                self._connection = self._create_connection()
            return self._connection

    def _create_connection(self) -> DriverRemoteConnection:
        """Open a connection and probe it with one request.

        Raises:
            ConfigurationError: Neptune refused the connection (HTTP 401/403).
            StoreConnectionError: Neptune could not be reached.
            AWSServiceError: The probe failed otherwise (a timeout, a
                throttled or failed request), which the next use may not.
        """
        if not self.neptune_config.endpoint:
            raise AWSServiceError("Neptune endpoint is not configured")

        scheme = "wss" if self.neptune_config.use_ssl else "ws"
        connection_url = (
            f"{scheme}://{self.neptune_config.endpoint}:"
            f"{self.neptune_config.port}/gremlin"
        )
        sign = self._get_auth_headers if self.neptune_config.use_iam else None
        timeout = self._timeout_seconds

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
            if isinstance(e, GraphRAGException):
                raise
            if is_connection_lost(e):
                raise StoreConnectionError(
                    f"Neptune is not reachable at '{connection_url}': {e}"
                ) from e
            raise AWSServiceError(f"Neptune connection probe failed: {e!r}") from e

    def _get_auth_headers(self, url: str) -> dict[str, str]:
        logger.debug("Using IAM authentication for Neptune connection")
        credentials = self.boto_session.get_credentials()
        if not credentials:
            raise AWSServiceError(
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

    @_handle_neptune_errors
    def delete_vertices_in_batches(self, label: str, batch_size: int = 500) -> None:
        """Drop every vertex of ``label``, ``batch_size`` per request.

        Bounded requests keep each drop's transaction small. A request returns
        how many vertices it dropped, so a short one is the last: no count of
        the label per pass and no pause between requests. The label is counted
        once up front to bound the passes: if requests keep reporting full
        batches past that bound (a backend whose count does not reflect the
        drop, or writers adding vertices meanwhile) the delete fails instead of
        looping forever.

        Raises:
            AWSServiceError: The label still reported full batches after
                ``ceil(count / batch_size) + _EXTRA_DELETE_PASSES`` passes.
        """
        initial = int(self.g.V().hasLabel(label).count().next())
        max_passes = math.ceil(initial / batch_size) + _EXTRA_DELETE_PASSES
        total = 0
        for _ in range(max_passes):
            dropped = int(
                self.g.V()
                .hasLabel(label)
                .limit(batch_size)
                .sideEffect(__.drop())
                .count()
                .next()
            )
            total += dropped
            if dropped < batch_size:
                break
        else:
            raise AWSServiceError(
                f"Deleting vertices with label '{label}' did not finish: "
                f"{max_passes} passes of {batch_size} each dropped a full batch "
                f"({total} reported dropped, {initial} counted before)"
            )
        logger.info("Deleted %s vertices with label '%s'", total, label)

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
