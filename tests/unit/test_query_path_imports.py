# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The query path keys entity names without loading entity resolution's
MinHash engine: datasketch and the SciPy it imports (about a second) load
only when a matcher needs them."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from unified_kg_rag.domain.ingestion import base_resolver, names

pytestmark = pytest.mark.unit


def test_the_query_path_imports_no_minhash_engine() -> None:
    script = (
        "import json, sys\n"
        "from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain\n"
        "import unified_kg_rag.adapters.storage.neptune_codec\n"
        "import unified_kg_rag.evaluation.graph_aware_evaluator\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules} "
        "& {'datasketch', 'scipy'})))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def test_the_resolver_reexports_the_name_keys() -> None:
    for name in names.__all__:
        assert getattr(base_resolver, name) is getattr(names, name)


def test_a_minhash_matcher_still_matches() -> None:
    matcher = base_resolver.FuzzyMatcher(
        ["Acme Logistics", "Globex Trading"],
        resolution_method="minhash",
        similarity_threshold=0.5,
    )
    assert [name for name, _ in matcher.find_all_matches("Acme Logistic")] == [
        "Acme Logistics"
    ]
