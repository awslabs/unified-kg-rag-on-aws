# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The search-engine surface ``OpenSearchIndexer`` and ``OpenSearchRetriever`` use.

``OpenSearchClient`` is the default implementation. Another engine that speaks
the same request/response shapes (the OpenSearch query DSL subset the indexer
and retriever build, and ``{"hits": {"hits": [...]}}`` responses) can be
injected instead, so both backends share one copy of the indexing and
retrieval logic. The method list is exactly what those two classes call;
``tests/unit/test_fake_port_signatures.py`` keeps ``OpenSearchClient``'s
signatures in step with it.
"""

from typing import Any, Protocol


class SearchEngineClient(Protocol):
    # --- writes -----------------------------------------------------------
    def bulk_index(
        self, index: str, documents: list[dict[str, Any]], refresh: bool = True
    ) -> dict[str, Any]: ...

    def bulk_delete(
        self, index: str, ids: list[str], refresh: bool = True
    ) -> dict[str, Any]: ...

    def bulk_update(
        self, index: str, documents: list[dict[str, Any]], refresh: bool = True
    ) -> dict[str, Any]: ...

    # --- index and alias administration -----------------------------------
    def create_index(
        self, index: str, body: dict[str, Any], *, exist_ok: bool = True
    ) -> None: ...

    def put_search_pipeline(self, pipeline_id: str, body: dict[str, Any]) -> None: ...

    def delete_alias(
        self, index_names: str | list[str], alias_names: str | list[str]
    ) -> None: ...

    def delete_indices(self, index_names: list[str]) -> None: ...

    def get_index_mappings(self, index_name: str) -> dict[str, Any]: ...

    def put_index_meta(self, index_name: str, meta: dict[str, Any]) -> None: ...

    def get_index_name_by_alias(self, alias_name: str) -> str | None: ...

    def get_indices_by_alias(self, alias_name: str) -> list[str]: ...

    def get_aliases_by_index(self, index_pattern: str) -> dict[str, list[str]]: ...

    def update_alias(
        self,
        alias_name: str,
        new_index_name: str,
        remove_pattern: str | None = None,
    ) -> None: ...

    # --- reads --------------------------------------------------------------
    def search(self, **kwargs: Any) -> dict[str, Any]: ...

    async def asearch(self, **kwargs: Any) -> dict[str, Any]: ...

    async def aget_mappings(self, index: str) -> dict[str, dict[str, Any]]: ...

    async def aindex_exists(self, index: str) -> bool: ...

    # --- lifecycle ----------------------------------------------------------
    def close(self) -> None: ...

    async def aclose(self) -> None: ...
