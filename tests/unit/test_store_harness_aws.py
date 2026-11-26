# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``aws`` store harness (tests/fixtures/store_backends.py) reaches a
managed domain as shipped: its config signs and uses TLS, and a suite
pointing its own config at the harness carries that connection over. No AWS
call: configs only."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.fixtures import store_backends
from tests.fixtures.store_backends import point_at_harness_stores
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.unit

NEPTUNE = "kg.cluster-example.us-west-2.neptune.amazonaws.com"
OPENSEARCH = "vpc-kg-example.us-west-2.es.amazonaws.com"


@pytest.fixture
def aws_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UKR_TEST_CONFIG", raising=False)
    for name in ("USERNAME", "PASSWORD", "USE_IAM", "ALLOW_ANONYMOUS"):
        monkeypatch.delenv(f"UKR__AWS__OPENSEARCH__{name}", raising=False)
    monkeypatch.setenv("UKR__AWS__NEPTUNE__ENDPOINT", NEPTUNE)
    monkeypatch.setenv("UKR__AWS__OPENSEARCH__ENDPOINT", OPENSEARCH)


def _assert_signed_tls(config: Config) -> None:
    neptune, opensearch = config.aws.neptune, config.aws.opensearch
    assert (neptune.endpoint, opensearch.endpoint) == (NEPTUNE, OPENSEARCH)
    assert neptune.use_iam and neptune.use_ssl
    assert opensearch.use_iam and opensearch.use_ssl and opensearch.verify_certs
    assert not opensearch.allow_anonymous
    assert opensearch.port == 443


def test_aws_harness_config_without_a_file_signs_and_uses_tls(
    aws_endpoints: None,
) -> None:
    # Without UKR_TEST_CONFIG the defaults left OpenSearch on basic auth with
    # no credentials: "No OpenSearch auth method configured" on any domain.
    config = store_backends._aws_config()
    _assert_signed_tls(config)
    assert (config.providers.graph.kind, config.providers.vector.kind) == (
        "neptune",
        "opensearch",
    )


def test_aws_harness_keeps_the_basic_auth_its_environment_names(
    aws_endpoints: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UKR__AWS__OPENSEARCH__USERNAME", "tester")
    monkeypatch.setenv("UKR__AWS__OPENSEARCH__PASSWORD", "synthetic")
    opensearch = store_backends._aws_config().aws.opensearch
    assert not opensearch.use_iam
    assert opensearch.username == "tester"


def test_a_local_preset_config_pointed_at_the_aws_harness_carries_its_connection(
    aws_endpoints: None,
) -> None:
    # The embedded suites start from a `backend: local` config (plain text,
    # unsigned, anonymous OpenSearch); copying only the endpoint and port
    # sent that to the domain, which refused it as unreachable.
    harness = SimpleNamespace(config=store_backends._aws_config())
    harness.config.aws.region_name = "eu-west-1"
    harness.config.indexing.additional_suffix = "cfsuffix"
    config = Config(backend="local")
    assert not config.aws.opensearch.use_ssl and config.aws.opensearch.allow_anonymous

    point_at_harness_stores(config, harness)  # type: ignore[arg-type]

    _assert_signed_tls(config)
    assert config.aws.region_name == "eu-west-1"
    assert (config.providers.graph.kind, config.providers.vector.kind) == (
        "neptune",
        "opensearch",
    )
    assert config.indexing.additional_suffix == "cfsuffix"
    # A copy: the suite changing its config leaves the harness's alone.
    config.aws.opensearch.port = 9200
    assert harness.config.aws.opensearch.port == 443
