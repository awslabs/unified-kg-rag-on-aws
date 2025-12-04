# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import asyncio
from collections.abc import Callable, Generator
from functools import wraps
from typing import Any

import boto3
from opensearchpy import (
    AIOHttpConnection,
    AsyncOpenSearch,
    OpenSearch,
    RequestsHttpConnection,
)
from opensearchpy.exceptions import NotFoundError, TransportError
from opensearchpy.helpers import streaming_bulk
from requests_aws4auth import AWS4Auth

from aws_graphrag.core import AWSServiceError, get_logger
from aws_graphrag.models import Config

logger = get_logger(__name__)


def _handle_opensearch_errors(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(client_instance: "OpenSearchClient", *args: Any, **kwargs: Any) -> Any:
        try:
            return func(client_instance, *args, **kwargs)
        except NotFoundError as e:
            logger.debug(f"Resource not found during '{func.__name__}': {e}")
            raise
        except TransportError as e:
            msg = f"Transport error in '{func.__name__}': status={e.status_code}, info={e.info}"
            logger.error(msg, exc_info=True)
            raise AWSServiceError(msg) from e
        except Exception as e:
            msg = f"Unexpected error in '{func.__name__}': {e}"
            logger.error(msg, exc_info=True)
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
            logger.debug(f"Resource not found during async '{func.__name__}': {e}")
            raise
        except TransportError as e:
            msg = f"Async transport error in '{func.__name__}': status={e.status_code}, info={e.info}"
            logger.error(msg, exc_info=True)
            raise AWSServiceError(msg) from e
        except Exception as e:
            msg = f"Unexpected async error in '{func.__name__}': {e}"
            logger.error(msg, exc_info=True)
            raise AWSServiceError(msg) from e

    return async_wrapper


class OpenSearchClient:
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
            self._async_client = self._create_async_client()
            self._bound_loop_id = current_loop_id

        return self._async_client

    def _get_current_loop_id(self) -> int | None:
        try:
            return id(asyncio.get_running_loop())
        except RuntimeError:
            return None

    def _create_client(self) -> OpenSearch:
        params = self._get_base_connection_params()
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
            logger.info(f"Connected to OpenSearch cluster: {cluster_name}")
            return client
        except Exception as e:
            logger.error(f"Failed to create OpenSearch client: {e}", exc_info=True)
            raise AWSServiceError("Failed to connect to OpenSearch.") from e

    def _create_async_client(self) -> AsyncOpenSearch:
        params = self._get_base_connection_params()
        params["connection_class"] = AIOHttpConnection
        return AsyncOpenSearch(**params)

    def _get_base_connection_params(self) -> dict[str, Any]:
        if not self.opensearch_config.endpoint:
            raise AWSServiceError("OpenSearch endpoint is not configured.")

        return {
            "hosts": [
                {
                    "host": self.opensearch_config.endpoint,
                    "port": self.opensearch_config.port,
                }
            ],
            "http_auth": self._get_auth(),
            "use_ssl": self.opensearch_config.use_ssl,
            "verify_certs": self.opensearch_config.verify_certs,
            "timeout": 180,
        }

    def _get_auth(self) -> AWS4Auth | tuple[str, str]:
        if self.opensearch_config.use_iam:
            creds = self.boto_session.get_credentials()
            if not creds:
                raise AWSServiceError("Cannot get AWS credentials for OpenSearch IAM.")
            return AWS4Auth(
                creds.access_key,
                creds.secret_key,
                self.config.aws.region_name,
                "es",
                session_token=creds.token,
            )

        if self.opensearch_config.username and self.opensearch_config.password:
            return self.opensearch_config.username, self.opensearch_config.password

        raise AWSServiceError("No OpenSearch auth method configured (IAM or basic).")

    @_handle_opensearch_errors
    def alias_exists(self, alias_name: str) -> bool:
        result = self.client.indices.exists_alias(name=alias_name)
        return bool(result)

    @_handle_opensearch_errors
    def bulk_index(
        self, index: str, documents: list[dict[str, Any]], refresh: bool = True
    ) -> dict[str, Any]:
        if not documents:
            return {"errors": False, "items": []}

        def generate_actions() -> Generator[dict[str, Any], None, None]:
            for doc in documents:
                action = {"_op_type": "index", "_index": index, "_source": doc}
                if (doc_id := doc.get("id")) is not None:
                    action["_id"] = str(doc_id)
                yield action

        success_count, errors = 0, []
        try:
            for ok, result in streaming_bulk(
                client=self.client, actions=generate_actions(), chunk_size=100
            ):
                if ok:
                    success_count += 1
                else:
                    errors.append(result)
        except Exception as e:
            raise AWSServiceError("Streaming bulk operation failed.") from e

        if errors:
            logger.warning(
                f"Bulk index completed with {len(errors)} errors in '{index}'"
            )
        else:
            logger.debug(f"Successfully indexed {success_count} documents to '{index}'")

        if refresh and success_count > 0:
            self.client.indices.refresh(index=index)

        return {"errors": bool(errors), "items": errors}

    @_handle_opensearch_errors
    def create_ingest_pipeline(self, pipeline_id: str, body: dict[str, Any]) -> None:
        if not self.check_ingest_pipeline_exists(pipeline_id):
            self.client.ingest.put_pipeline(id=pipeline_id, body=body)
            logger.debug(f"Created ingest pipeline: '{pipeline_id}'")

    @_handle_opensearch_errors
    def check_ingest_pipeline_exists(self, pipeline_id: str) -> bool:
        try:
            self.client.ingest.get_pipeline(id=pipeline_id)
            return True
        except NotFoundError:
            return False

    @_handle_opensearch_errors
    def create_index(self, index: str, body: dict[str, Any]) -> None:
        if not self.client.indices.exists(index=index):
            self.client.indices.create(index=index, body=body)
            logger.info(f"Created index: '{index}'")

    @_handle_opensearch_errors
    def create_search_pipeline(self, pipeline_id: str, body: dict[str, Any]) -> None:
        if not self.check_search_pipeline_exists(pipeline_id):
            self.client.search_pipeline.put(id=pipeline_id, body=body)
            logger.debug(f"Created search pipeline: '{pipeline_id}'")

    @_handle_opensearch_errors
    def check_search_pipeline_exists(self, pipeline_id: str) -> bool:
        try:
            self.client.search_pipeline.get(id=pipeline_id)
            return True
        except NotFoundError:
            return False

    @_handle_opensearch_errors
    def count(
        self, index_names: str | list[str], query: dict[str, Any] | None = None
    ) -> int:
        final_indices = (
            ",".join(index_names) if isinstance(index_names, list) else index_names
        )

        params: dict[str, Any] = {"index": final_indices}
        if query:
            params["body"] = query

        logger.debug(f"Executing count on indices '{final_indices}'")
        response = self.client.count(**params)
        return int(response.get("count", 0))

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
            f"Deleting alias(es) '{final_aliases}' from index(es) '{final_indices}'"
        )
        self.client.indices.delete_alias(index=final_indices, name=final_aliases)

    @_handle_opensearch_errors
    def delete_indices(self, index_names: list[str]) -> None:
        if index_names:
            indices_str = ",".join(index_names)
            self.client.indices.delete(index=indices_str)
            logger.info(f"Deleted indices: {indices_str}")

    @_handle_opensearch_errors
    def delete_ingest_pipeline(self, pipeline_id: str) -> None:
        if self.check_ingest_pipeline_exists(pipeline_id):
            self.client.ingest.delete_pipeline(id=pipeline_id)
            logger.debug(f"Deleted ingest pipeline: '{pipeline_id}'")

    @_handle_opensearch_errors
    def delete_search_pipeline(self, pipeline_id: str) -> None:
        if self.check_search_pipeline_exists(pipeline_id):
            self.client.search_pipeline.delete(id=pipeline_id)
            logger.debug(f"Deleted search pipeline: '{pipeline_id}'")

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
    def get_index_stats(self, index_patterns: str | list[str]) -> dict[str, Any]:
        target = (
            ",".join(index_patterns)
            if isinstance(index_patterns, list)
            else index_patterns
        )
        try:
            result = self.client.indices.stats(index=target, metric="_all")
            return dict(result)
        except NotFoundError:
            return {}

    @_handle_opensearch_errors
    def search(self, **kwargs: Any) -> dict[str, Any]:
        result = self.client.search(**kwargs)
        return dict(result)

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
        logger.info(f"Updated alias '{alias_name}' to point to '{new_index_name}'")
