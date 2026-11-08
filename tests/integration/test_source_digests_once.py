# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run reads each source file's bytes for its digests once (AWS-free).

The corpus fingerprint hashed every file, then the parsing stage read each
again for its source digest; the fingerprint's read now computes both.
"""

from __future__ import annotations

import pathlib
from collections import Counter
from pathlib import Path

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - a fixture
from unified_kg_rag.application.ingestion import source_parsing
from unified_kg_rag.domain.models import PipelineStageStatus

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]


def test_a_run_reads_each_source_file_once_for_its_digests(
    tmp_path: Path, mocker
) -> None:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    providers = e2e._providers(
        config, FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    source_parsing._SOURCE_DIGESTS.clear()
    opened: Counter[str] = Counter()
    open_ = pathlib.Path.open

    def counted(
        self, mode="r", *args, **kwargs
    ):  # noqa: ANN001, ANN002, ANN003, ANN202
        if mode == "rb" and self.parent == source:
            opened[self.name] += 1
        return open_(self, mode, *args, **kwargs)

    mocker.patch.object(pathlib.Path, "open", counted)

    context = e2e._ingest(config, providers, source, tmp_path / "run-1")

    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert opened == {"contract.txt": 1, "logistics.txt": 1}
