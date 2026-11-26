# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``aws`` store harness (tests/fixtures/store_backends.py) reaches a
managed domain as shipped: its config signs and uses TLS, a suite pointing
its own config at the harness carries that connection over, and its teardown
outlasts the domain's automated snapshot. No AWS call: configs and fakes
only."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from opensearchpy.exceptions import RequestError

from tests.fixtures import store_backends
from tests.fixtures.store_backends import (
    SNAPSHOT_RETRY_DELAYS_SECONDS,
    SNAPSHOT_WAIT_SECONDS,
    SnapshotWait,
    StoreHarness,
    clear_through_snapshots,
    point_at_harness_stores,
)
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared.exceptions import AWSServiceError

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


def _snapshot_refusal() -> AWSServiceError:
    """What ``OpenSearchClient.delete_indices`` raises while the domain's
    automated snapshot runs."""
    cause = RequestError(
        400,
        "snapshot_in_progress_exception",
        {"error": {"type": "snapshot_in_progress_exception"}},
    )
    error = AWSServiceError.caused_by(
        "Failed to delete 1 of 1 index name(s): kg-text-units; first error", cause
    )
    error.__cause__ = cause
    return error


class _Clock:
    """A monotonic clock that only ``sleep`` (or a clear that takes time)
    moves, recording each sleep."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def wait(self) -> SnapshotWait:
        return SnapshotWait(clock=self, sleep=self.sleep)


class _Refusing:
    """A ``clear`` the domain refuses ``refusals`` times, then runs."""

    def __init__(self, refusals: int, error: Exception | None = None) -> None:
        self.refusals = refusals
        self.error = error
        self.calls: list[list[str]] = []

    def __call__(self, suffixes: list[str]) -> None:
        self.calls.append(suffixes)
        if len(self.calls) <= self.refusals:
            raise self.error or _snapshot_refusal()


def test_a_clear_refused_for_a_snapshot_runs_again_after_a_backoff() -> None:
    clear, clock = _Refusing(2), _Clock()
    clear_through_snapshots(clear, ["default"], clock.wait())
    assert clear.calls == [["default"]] * 3
    assert clock.sleeps == list(SNAPSHOT_RETRY_DELAYS_SECONDS[:2])


def test_a_snapshot_outlasting_the_wait_fails_the_clear() -> None:
    clear, clock = _Refusing(100), _Clock()
    with pytest.raises(AWSServiceError, match="Failed to delete"):
        clear_through_snapshots(clear, ["default"], clock.wait())
    assert clock.sleeps == list(SNAPSHOT_RETRY_DELAYS_SECONDS)
    assert sum(clock.sleeps) == SNAPSHOT_WAIT_SECONDS < 120 / 2  # the test timeout


def test_the_clears_of_one_teardown_share_one_wait() -> None:
    # r11diff-5: the bound was per clear, so a teardown of several clears
    # (3 per harness) could wait 30 s for each and outlast the test timeout.
    clock = _Clock()
    wait = clock.wait()
    clear_through_snapshots(_Refusing(2), ["default"], wait)
    with pytest.raises(AWSServiceError):
        clear_through_snapshots(_Refusing(100), ["default"], wait)
    # 2 + 4, then 2 + 4 + 8 and the 10 s left of the next 16.
    assert clock.sleeps == [2.0, 4.0, 2.0, 4.0, 8.0, 10.0]
    assert sum(clock.sleeps) == SNAPSHOT_WAIT_SECONDS
    # Spent: a later refusal fails at once.
    late = _Refusing(1)
    with pytest.raises(AWSServiceError):
        clear_through_snapshots(late, ["default"], wait)
    assert len(late.calls) == 1 and len(clock.sleeps) == 6


def test_the_clears_own_time_counts_against_the_wait() -> None:
    clock = _Clock()
    wait = clock.wait()

    def slow_refusal(suffixes: list[str]) -> None:
        clock.now += SNAPSHOT_WAIT_SECONDS
        raise _snapshot_refusal()

    with pytest.raises(AWSServiceError):
        clear_through_snapshots(slow_refusal, ["default"], wait)
    assert clock.sleeps == []


def test_any_other_clear_error_is_raised_at_once() -> None:
    clear, clock = _Refusing(1, AWSServiceError("access denied")), _Clock()
    with pytest.raises(AWSServiceError, match="access denied"):
        clear_through_snapshots(clear, ["default"], clock.wait())
    assert len(clear.calls) == 1 and clock.sleeps == []


class _FakeClient:
    """The OpenSearch client calls the teardown makes: no extraction-records
    index left, and deletes the domain refuses ``refusals`` times."""

    def __init__(self, refusals: int) -> None:
        self.delete_indices = _Refusing(refusals)

    def get_aliases_by_index(self, pattern: str) -> dict[str, Any]:
        return {}


class _FakeVectorStore:
    def __init__(self, client: _FakeClient) -> None:
        self.opensearch_client = client
        self.opensearch_config = SimpleNamespace(
            extraction_records_index_prefix="kg-extraction-records"
        )
        self.closed = False

    def _get_name(self, prefix: str, suffix: str) -> str:
        return f"{prefix}-{suffix}"

    def clear(self, suffixes: list[str]) -> None:
        self.opensearch_client.delete_indices(
            [self._get_name("kg-text-units", s) for s in suffixes]
        )

    def close(self) -> None:
        self.closed = True


class _FakeGraphStore:
    def clear(self, suffixes: list[str]) -> None:
        pass

    def close(self) -> None:
        pass


def _aws_harness(
    refusals: int, parent: StoreHarness | None = None
) -> tuple[StoreHarness, _FakeClient]:
    harness = StoreHarness("fake", Config(), parent=parent)
    harness.kind = "aws"
    client = _FakeClient(refusals)
    harness.graph_indexer = _FakeGraphStore()
    harness.vector_indexer = _FakeVectorStore(client)
    return harness, client


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    clock = _Clock()
    monkeypatch.setattr(store_backends.time, "monotonic", clock)
    monkeypatch.setattr(store_backends.time, "sleep", clock.sleep)
    return clock


def test_the_aws_teardown_outlasts_an_automated_snapshot(clock: _Clock) -> None:
    harness, client = _aws_harness(refusals=2)
    harness.teardown()
    assert len(client.delete_indices.calls) == 3
    assert clock.sleeps == list(SNAPSHOT_RETRY_DELAYS_SECONDS[:2])
    assert harness.vector_indexer.closed


def test_the_aws_teardown_reports_a_snapshot_that_does_not_end(clock: _Clock) -> None:
    harness, _ = _aws_harness(refusals=100)
    with pytest.raises(AssertionError, match="left data behind: vector clear failed"):
        harness.teardown()


def test_the_aws_teardown_waits_once_for_every_harness(clock: _Clock) -> None:
    # r11diff-5: the root and its derived harnesses are cleared in one
    # teardown, so their waits share one deadline.
    root, _ = _aws_harness(refusals=100)
    derived, derived_client = _aws_harness(refusals=100, parent=root)
    with pytest.raises(
        AssertionError, match="vector clear failed.*vector clear failed"
    ):
        root.teardown()
    assert sum(clock.sleeps) == SNAPSHOT_WAIT_SECONDS
    assert len(derived_client.delete_indices.calls) == 1
    assert root.vector_indexer.closed and derived.vector_indexer.closed
