# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run reads each source file's bytes for its digests once (AWS-free).

The corpus fingerprint hashed every file, then the parsing stage read each
again for its source digest; the fingerprint's read now computes both.
"""

from __future__ import annotations

import os
import pathlib
import time
from collections import Counter
from pathlib import Path

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - a fixture
from unified_kg_rag.application.ingestion import source_parsing
from unified_kg_rag.application.ingestion.source_parsing import (
    SourceFileParser,
    fingerprint_digest,
)
from unified_kg_rag.domain.models import Config, PipelineStageStatus

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


def test_a_file_rewritten_at_the_same_size_and_time_is_read_again(
    tmp_path: Path,
) -> None:
    # A sync that copies an object's second-precision time can rewrite a
    # file at the same size and modification time; the digest the
    # fingerprint kept must not stand for the new bytes.
    path = tmp_path / "contract.txt"
    path.write_text("Vendor supplies widgets to Buyer.")
    stat = path.stat()
    fingerprint_digest(path)
    parser = SourceFileParser(Config(), tmp_path)
    before = parser.source_digest(path)
    time.sleep(0.01)
    path.write_text("Vendor supplies gadgets to Buyer.")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_size == stat.st_size
    assert path.stat().st_mtime_ns == stat.st_mtime_ns

    assert parser.source_digest(path) != before


def test_a_run_with_an_encoding_reads_each_source_file_once(
    tmp_path: Path, mocker
) -> None:
    # rv26 algo-index perf: with processing.document_parsing.encoding set,
    # the digest the corpus fingerprint kept (keyed with no encoding) was
    # skipped and every file read twice. The cache keys the encoding.
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.processing.document_parsing.encoding = "cp1252"
    providers = e2e._providers(
        config, FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "legacy.txt").write_bytes(
        "Vendor’s café supplies Depot.".encode("cp1252")
    )
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
    assert opened == {"contract.txt": 1, "legacy.txt": 1}


def test_the_kept_digest_is_the_one_its_encoding_gives(tmp_path: Path) -> None:
    # A digest kept without the encoding is not served to a parser that
    # configures one, nor the other way round.
    path = tmp_path / "legacy.txt"
    path.write_bytes("Vendor’s café".encode("cp1252"))
    plain = SourceFileParser(Config(), tmp_path)
    cp1252 = SourceFileParser(
        Config(processing={"document_parsing": {"encoding": "cp1252"}}), tmp_path
    )
    source_parsing._SOURCE_DIGESTS.clear()
    want = (plain.source_digest(path), cp1252.source_digest(path))
    assert want[0] != want[1]
    for config in (Config(), cp1252.config):
        source_parsing._SOURCE_DIGESTS.clear()
        fingerprint_digest(path, config)
        assert (plain.source_digest(path), cp1252.source_digest(path)) == want
