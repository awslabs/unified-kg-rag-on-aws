import functools
from collections.abc import Callable
from typing import Any

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from gremlin_python.driver.driver_remote_connection import DriverRemoteConnection
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.graph_traversal import GraphTraversal, GraphTraversalSource

from aws_graphrag.core import AWSServiceError, get_logger
from aws_graphrag.models import Config

logger = get_logger(__name__)


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
        logger.debug("Neptune client initialized")

    @property
    def g(self) -> GraphTraversalSource:
        if self._g is None or self.connection.is_closed():
            self._g = traversal().withRemote(self.connection)
        return self._g

    @property
    def connection(self) -> DriverRemoteConnection:
        if self._connection is None or self._connection.is_closed():
            logger.debug("Establishing new Neptune connection")
            self._connection = self._create_connection()
        return self._connection

    def _create_connection(self) -> DriverRemoteConnection:
        if not self.neptune_config.endpoint:
            raise AWSServiceError("Neptune endpoint is not configured")

        connection_url = (
            f"wss://{self.neptune_config.endpoint}:{self.neptune_config.port}/gremlin"
        )
        headers = (
            self._get_auth_headers(connection_url)
            if self.neptune_config.use_iam
            else {}
        )

        try:
            remote_connection = DriverRemoteConnection(
                url=connection_url, traversal_source="g", headers=headers
            )
            g = traversal().withRemote(remote_connection)
            g.V().limit(1).toList()
            logger.info(
                f"Successfully connected to Neptune at '{self.neptune_config.endpoint}'"
            )
            return remote_connection
        except Exception as e:
            error_message = (
                f"Failed to establish connection to Neptune at '{connection_url}': {e}"
            )
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
    def clear_graph(self) -> None:
        logger.info("Clearing Neptune graph database")
        self.g.V().drop().iterate()
        logger.info("Successfully cleared Neptune graph")

    def close(self) -> None:
        if self._connection and not self._connection.is_closed():
            try:
                self._connection.close()
                self._connection = None
                self._g = None
                logger.info("Closed Neptune connection")
            except Exception as e:
                logger.error(f"Error closing Neptune connection: {e}")

    @_handle_neptune_errors
    def get_graph_stats(self) -> dict[str, Any]:
        logger.debug("Retrieving Neptune graph statistics")
        stats = {
            "vertex_count": self.g.V().count().next(),
            "edge_count": self.g.E().count().next(),
            "vertex_labels": self.g.V().label().dedup().toList(),
            "edge_labels": self.g.E().label().dedup().toList(),
        }
        logger.info(
            f"Graph stats: {stats['vertex_count']} vertices, {stats['edge_count']} edges"
        )
        return stats

    @_handle_neptune_errors
    def submit(self, traversal_query: GraphTraversal) -> list[Any] | Any:
        try:
            terminating_steps = {"count", "next", "head"}
            has_terminating_step = any(
                step[0] in terminating_steps
                for step in traversal_query.bytecode.step_instructions
            )
            return (
                traversal_query.next()
                if has_terminating_step
                else traversal_query.to_list()
            )
        except Exception as e:
            logger.error(f"Failed to execute traversal: {e}", exc_info=True)
            raise AWSServiceError(f"Traversal execution failed: {e}") from e
