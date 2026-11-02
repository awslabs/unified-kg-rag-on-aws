# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A seed graph's properties through incremental runs on the embedded stores
(see ``test_seed_graph_embedded``): a second seed that states only a
document entity's properties, then is removed, leaves no property behind in
either store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - fixture
from tests.integration.test_seed_graph_embedded import SHARDS, _Harness
from unified_kg_rag.domain.models.config import SeedGraphConfig

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]


def test_a_removed_seed_takes_its_properties_out_of_both_stores(
    tmp_path: Path,
) -> None:
    h = _Harness(tmp_path)
    beta = tmp_path / "beta.jsonl"
    beta.write_text(
        json.dumps(
            {
                "kind": "node",
                "id": "v",
                "name": "Vendor",
                "properties": {"tier": "gold", "": "odd key", "n": 2**64},
            }
        )
        + "\n"
    )
    procure = SeedGraphConfig(
        name="procure", kind="jsonl", path=str(h.seed), shards=SHARDS
    )
    h.config.processing.seed_graphs = [
        procure,
        SeedGraphConfig(name="beta", kind="jsonl", path=str(beta), shards=1),
    ]
    h.ok()
    vendor = h.entities()["Vendor"]
    assert vendor["attributes"]["seed_properties"] == {
        "tier": "gold",
        "": "odd key",
        "n": 2**64,
    }

    # Nothing else of Vendor changes: procure still states it.
    h.config.processing.seed_graphs = [procure]
    h.ok()
    vendor = h.entities()["Vendor"]
    assert "seed_properties" not in (vendor.get("attributes") or {})
    graph = h.providers.build_graph_indexer()
    try:
        (stored,) = graph.read_entities([vendor["id"]])
    finally:
        graph.close()
    assert "seed_properties" not in (stored.attributes or {})
