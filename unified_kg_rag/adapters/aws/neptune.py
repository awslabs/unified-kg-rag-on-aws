# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import threading
import time
from collections.abc import Callable
from typing import Any

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.driver.protocol import GremlinServerError
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import GraphTraversalSource

from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import AWSServiceError, get_logger

logger = get_logger(__name__)

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
    def __init__(self, config: Config, boto_session: boto3.Session | None = None):
        self.config = config
        self.neptune_config = config.aws.neptune
        self.boto_session = boto_session or boto3.Session(
            profile_name=config.aws.profile_name,
            region_name=config.aws.region_name,
        )
        self._connection: DriverRemoteConnection | None = None
        self._g: GraphTraversalSource | None = None
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        logger.debug("Neptune client initialized")

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
        if not self.neptune_config.endpoint:
            raise AWSServiceError("Neptune endpoint is not configured")

        scheme = "wss" if self.neptune_config.use_ssl else "ws"
        connection_url = (
            f"{scheme}://{self.neptune_config.endpoint}:"
            f"{self.neptune_config.port}/gremlin"
        )
        headers = (
            self._get_auth_headers(connection_url)
            if self.neptune_config.use_iam
            else {}
        )

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
                headers=headers,
                pool_size=pool_size,
                max_workers=pool_size,
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
            error_message = f"Failed to establish connection to Neptune: {e}"
            logger.error(error_message)
            raise AWSServiceError(error_message) from e

    def _get_auth_headers(self, url: str) -> dict[str, str]:
        logger.debug("Using IAM authentication for Neptune connection")
        credentials = self.boto_session.get_credentials()
        if not credentials:
            raise AWSServiceError(
                "Unable to get AWS credentials for IAM authentication"
            )

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
    def delete_vertices_in_batches(
        self, label: str, batch_size: int = 500, delay: float = 0.5
    ) -> None:
        logger.info(
            "Starting batch deletion for label '%s' with batch size %s.",
            label,
            batch_size,
        )
        previous_count: int | None = None
        while True:
            remaining_count = self.g.V().hasLabel(label).count().next()
            if remaining_count == 0:
                logger.info("No more vertices with label '%s' to delete.", label)
                break

            # Guard against an infinite loop: if a drop pass does not reduce the
            # count (e.g. undeletable vertices, or a stale count), stop rather
            # than spin forever sleeping `delay` each pass.
            if previous_count is not None and remaining_count >= previous_count:
                raise AWSServiceError(
                    f"Batch deletion for label '{label}' made no progress "
                    f"({remaining_count} vertices remain); aborting to avoid an "
                    "infinite loop."
                )
            previous_count = remaining_count

            logger.info(
                "Deleting batch of %s from %s vertices with label '%s'...",
                min(remaining_count, batch_size),
                remaining_count,
                label,
            )
            self.g.V().hasLabel(label).limit(batch_size).drop().iterate()
            time.sleep(delay)
        logger.info("Finished batch deletion for label '%s'.", label)

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
