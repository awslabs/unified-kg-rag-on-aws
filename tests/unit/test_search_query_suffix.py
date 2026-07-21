# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SearchQuery.suffix must be validated on the read path (index-name injection).

Regression: the write side validated the suffix, but the query-path suffix flowed
verbatim into the OpenSearch index target (f"{prefix}-{suffix}"), enabling
index-name injection / cross-tenant reads (e.g. "*", comma-joined aliases).
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.models import SearchQuery

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("good", [None, "default", "tenant-a", "v2_1", "abc-123"])
def test_valid_suffixes_accepted(good) -> None:
    assert SearchQuery(query="q", suffix=good).suffix == good


@pytest.mark.parametrize(
    "bad",
    ["*", "a,b", "Tenant", "other/tenant", "a b", "évil", "..", "a*"],
)
def test_injection_suffixes_rejected(bad) -> None:
    with pytest.raises(ValueError):
        SearchQuery(query="q", suffix=bad)
