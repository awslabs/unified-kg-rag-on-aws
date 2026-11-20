# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every stored text unit of a namespace, for tests that check what a run
stored (the stores page them: ``SummaryTreeStore.scan_text_unit_pages``)."""

from __future__ import annotations

from typing import Any

from unified_kg_rag.ports.indexer import SummaryTreeStore


def stored_text_units(
    store: SummaryTreeStore, suffix: str | None, *, fields: list[str]
) -> list[dict[str, Any]]:
    """The stored text units of ``suffix`` with ``fields``, in id order;
    ``[]`` when the store cannot page them."""
    pages = store.scan_text_unit_pages(suffix, fields=fields)
    return [unit for page in pages or [] for unit in page]
