# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property-based tests for content hashing.

The DocStatusPort.diff partition properties run against every registry in
``tests/integration/test_doc_status_conformance.py``.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from unified_kg_rag.shared.utils.common import compute_hash

pytestmark = pytest.mark.property


@given(st.text(), st.integers(min_value=1, max_value=64))
def test_compute_hash_length_is_bounded(data: str, length: int) -> None:
    assert len(compute_hash(data, length=length)) == length
