# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The forkserver preloads every pool module of a run, whichever pool
starts it."""

from __future__ import annotations

import json
import multiprocessing
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit

_PROBE = """
import json
import multiprocessing

import unified_kg_rag.application.ingestion.pipeline  # noqa: F401 - the stages
from unified_kg_rag.shared.utils.concurrency import process_pool

seen = []
context = multiprocessing.get_context("forkserver")
context.set_forkserver_preload = lambda modules: seen.append(list(modules))
multiprocessing.get_context = lambda method=None: context
pool = process_pool(1, preload=("unified_kg_rag.adapters.ingestion.loader",))
pool.shutdown()
print(json.dumps(seen))
"""


@pytest.mark.skipif(
    "forkserver" not in multiprocessing.get_all_start_methods(),
    reason="no forkserver on this platform",
)
def test_the_first_pool_preloads_the_other_pools_modules() -> None:
    # The forkserver starts with the first pool and reads its preload list
    # once: the loader's pool started it with the loader alone, so graph
    # resolution's workers each imported the resolver modules again.
    done = subprocess.run(
        [sys.executable, "-c", _PROBE], capture_output=True, text=True, check=True
    )
    (modules,) = json.loads(done.stdout.strip().splitlines()[-1])
    assert {
        "unified_kg_rag.adapters.ingestion.loader",
        "unified_kg_rag.domain.ingestion.graph_resolver",
    } <= set(modules)
