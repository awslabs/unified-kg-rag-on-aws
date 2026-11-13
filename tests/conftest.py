# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest fixtures.

Tests run AWS-free by default: the ``aws`` marker gates the few that need real
services, and port-based fakes (``tests/fixtures/fakes``) stand in for Neptune /
OpenSearch / DynamoDB. ``moto`` provides mocked AWS APIs where an adapter must
be exercised against a boto3 surface.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import settings
from langsmith.utils import get_env_var

from unified_kg_rag.adapters.aws.bedrock import BedrockCrossRegionModelHelper
from unified_kg_rag.domain.models import Config

# Hypothesis profiles, selected with HYPOTHESIS_PROFILE (default: "default").
# "ci" removes the per-example deadline, whose wall-clock limit flakes on
# shared runners, and derandomizes generation so a failure reproduces on
# re-run; print_blob makes any failure replayable locally via @reproduce_failure.
settings.register_profile("ci", deadline=None, derandomize=True, print_blob=True)
# "explore" (the weekly hypothesis-explore workflow) draws from a fresh random
# seed each run and raises the example budget of the tests that do not set
# their own; a test with its own max_examples keeps it, but still meets new
# examples. A failure prints the blob that replays it.
settings.register_profile("explore", max_examples=1000, deadline=None, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

# The suite directory a test lives under names its kind; its marker is applied
# here, so a new test file cannot miss it (``-m unit`` / ``-m "not property"``).
_SUITE_MARKERS = ("unit", "integration", "property")
_TESTS_ROOT = Path(__file__).parent

_PROFILE_PREFIXES = ("global", "us", "eu", "apac")

# Ambient AWS settings that would make boto3 resolve the developer's real
# credentials/profile (or fail with ProfileNotFound when it does not exist).
_AWS_ENV_TO_CLEAR = (
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
    "AWS_REGION",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
)
# Every UKR_ variable (UKR__<path> config overrides and the UKR_ run-ingestion
# defaults) and OpenTelemetry's own (OTEL_SDK_DISABLED turns tracing off,
# OTEL_EXPORTER_* points it elsewhere) is cleared too (see _isolated_aws_env),
# so a value exported in the developer's shell cannot silently change what
# the tests exercise. UKR_TEST_* configure the test run itself and stay.
_ENV_PREFIXES_TO_CLEAR = ("UKR_", "OTEL_")
_TEST_ENV_PREFIX = "UKR_TEST_"
# Other environment the package reads: run-ingestion argument defaults, and
# LangSmith tracing (every name langsmith accepts), which would upload test
# prompts and outputs and log a warning from every setup_logging() call.
_OTHER_ENV_TO_CLEAR = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
)
_FAKE_AWS_ENV = {
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",  # nosec B105 - placeholder, not a secret
    "AWS_DEFAULT_REGION": "us-east-1",
    "AWS_CONFIG_FILE": os.devnull,
    "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
    "AWS_EC2_METADATA_DISABLED": "true",
}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark each test with its suite (``tests/<suite>/``) before ``-m``
    selects: this runs ahead of the built-in deselection hook."""
    for item in items:
        try:
            suite = item.path.relative_to(_TESTS_ROOT).parts[0]
        except ValueError:
            continue
        if suite in _SUITE_MARKERS and item.get_closest_marker(suite) is None:
            item.add_marker(suite)


@pytest.fixture(autouse=True, scope="session")
def _thread_only_tqdm_lock() -> None:
    """Give tqdm a plain thread lock for the session.

    tqdm's default write lock pairs a threading RLock with a multiprocessing
    one, a semaphore shared with every process forked from this one (the
    tests' real process pools) and acquired in two steps. A worker once hung
    for the whole test timeout in ``tqdm.__new__`` with no live thread
    holding the lock (test_chain_factory_output_guards, under load), the
    shape of a lock left held across a fork or by an exception between the
    two steps. The tests draw no bars across processes, so one atomic thread
    lock is enough.
    """
    import threading

    from tqdm import tqdm

    tqdm.set_lock(threading.RLock())


@pytest.fixture(autouse=True, scope="session")
def _scratch_working_directory(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[None]:
    """Run the session in a scratch working directory.

    The CLIs and stages resolve their relative defaults (``logs/``,
    ``outputs/visualization``, ``cache/``) against the working directory, so
    tests that keep those defaults would otherwise write into the checkout;
    a file handler a CLI test attaches keeps logging every later test there.
    Tests read repository files relative to ``__file__``, never the CWD.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(tmp_path_factory.mktemp("cwd"))
        yield


@pytest.fixture(autouse=True)
def _no_takeover_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """A takeover of an expired lease waits up to ~3 minutes for the late
    writes of its holder (``namespace_lease``), and a stop leaves its lease
    expired: tests that stop a run and run again need no such wait. The
    tests of the wait itself set a short bound."""
    from unified_kg_rag.application.ingestion import namespace_lease

    monkeypatch.setattr(namespace_lease, "longest_write_request_seconds", lambda: 0.0)
    monkeypatch.setattr(namespace_lease, "_TAKEOVER_MARGIN_SECONDS", 0.0)


@pytest.fixture(autouse=True)
def _cli_stops_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stopped CLI raises its SystemExit instead of ending the process
    (``conventions._exit_stopped``), so the suite survives it."""
    from unified_kg_rag.application.cli import conventions

    monkeypatch.setitem(conventions._exit_at_once, "enabled", False)


@pytest.fixture(autouse=True)
def _restored_console(monkeypatch: pytest.MonkeyPatch) -> None:
    """A CLI run with ``--output-format json`` sends the shared console to
    stderr for the rest of the process (``conventions.exits_on_sigterm``);
    a test that runs one in-process would otherwise leave every later
    test's console output on stderr."""
    from unified_kg_rag.application.cli.display import console

    monkeypatch.setattr(console, "stderr", console.stderr)


@pytest.fixture(autouse=True)
def _isolated_logging_setup() -> Iterator[None]:
    """Undo a ``setup_logging`` call (a CLI run in-process) after each test.

    While ``LoggingSetup._handlers`` is non-empty every structlog record is
    handed to stdlib as the event dict for those handlers to render
    (``shared.logging._to_stdlib``), so a later test's ``caplog`` would see
    a dict as ``msg`` instead of the message. The call also replaces the
    structlog configuration and sets the root, package and library logger
    levels; all of it is restored here.
    """
    import logging

    import structlog

    from unified_kg_rag.shared.logging import LoggingSetup

    root = logging.getLogger()
    saved_handlers = list(LoggingSetup._handlers)
    saved_structlog = structlog.get_config() if structlog.is_configured() else None
    names = ("", "unified_kg_rag", *Config().logging.library_levels)
    saved_levels = {name: logging.getLogger(name).level for name in names}
    yield
    for handler in LoggingSetup._handlers:
        if handler not in saved_handlers:
            root.removeHandler(handler)
            handler.close()
    LoggingSetup._handlers = saved_handlers
    # Unconfigured before (nothing logged yet) stays as the test left it:
    # structlog's defaults would send a not-yet-used module logger to print.
    if saved_structlog is not None:
        structlog.configure(**saved_structlog)
    for name, level in saved_levels.items():
        logging.getLogger(name).setLevel(level)


@pytest.fixture(autouse=True)
def _isolated_aws_env(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin every test to fake, local-only AWS credentials and config.

    Also clears the other environment the package reads (config overrides,
    CLI defaults, LangSmith tracing) so the developer's shell cannot change
    what a test exercises.

    Without this the suite inherits the developer's shell: a non-existent
    ``AWS_PROFILE`` makes boto3 raise ``ProfileNotFound`` in hundreds of tests,
    and a valid one (or an instance role via IMDS) lets an accidentally
    unmocked client reach a real account. Config files point at ``/dev/null``
    and IMDS is disabled so no credential source other than the fake keys is
    consulted. Tests that need a specific value set it with ``monkeypatch``.
    Tests marked ``aws`` keep the real environment, as they keep the real
    transport (``_block_real_aws_http``): they read the store endpoints from
    it and call AWS with the real credential chain.
    """
    if request.node.get_closest_marker("aws") is not None:
        return
    for name in _AWS_ENV_TO_CLEAR + _OTHER_ENV_TO_CLEAR:
        monkeypatch.delenv(name, raising=False)
    for name in list(os.environ):
        upper = name.upper()
        if upper.startswith(_ENV_PREFIXES_TO_CLEAR) and not upper.startswith(
            _TEST_ENV_PREFIX
        ):
            monkeypatch.delenv(name, raising=False)
    for name, value in _FAKE_AWS_ENV.items():
        monkeypatch.setenv(name, value)
    get_env_var.cache_clear()  # langsmith caches its environment lookups


class RealAWSCallBlocked(RuntimeError):
    """Raised when a non-``aws`` test lets a botocore request reach the wire."""


@pytest.fixture(autouse=True)
def _block_real_aws_http(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail an unmocked botocore HTTP send immediately instead of dialling AWS.

    botocore treats connection errors as retryable, so an unmocked client used
    to cost ~20 s of backoff per test offline and silently succeeded (or was
    denied) against a real account online. A plain ``RuntimeError`` subclass is
    not retried, so the code under test sees one fast, deterministic failure.
    ``moto`` intercepts requests before this transport layer and is unaffected;
    tests marked ``aws`` keep the real transport.
    """
    if request.node.get_closest_marker("aws") is not None:
        return
    from botocore.httpsession import URLLib3Session

    def _blocked_send(self: URLLib3Session, aws_request: object) -> object:
        url = getattr(aws_request, "url", "<unknown>")
        raise RealAWSCallBlocked(f"real AWS HTTP call blocked in tests: {url}")

    monkeypatch.setattr(URLLib3Session, "send", _blocked_send)


_OFFLINE_REGIONS = ("us-east-1", "us-west-2", "eu-west-1", "ap-northeast-2")


@pytest.fixture(autouse=True)
def _offline_inference_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve Bedrock inference profiles from a static set, never from AWS.

    Profile resolution otherwise calls ``bedrock:ListInferenceProfiles``. That
    call succeeds on a developer machine with credentials and fails in CI, so
    profile-only models (e.g. Claude Haiku 4.5) behaved differently between the
    two. Tests that exercise the lookup itself clear this cache and stub the
    client explicitly.
    """
    from unified_kg_rag.domain.models import LanguageModelId

    profiles = {
        f"{prefix}.{model.value}"
        for model in LanguageModelId
        for prefix in _PROFILE_PREFIXES
    }
    monkeypatch.setattr(
        BedrockCrossRegionModelHelper,
        "_profiles_by_region",
        {region: set(profiles) for region in _OFFLINE_REGIONS},
    )


@pytest.fixture
def restore_renderer_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undo renderers a test registers with ``@register_renderer``.

    The registry is module-level, so a test-only renderer would otherwise stay
    registered for the rest of the session.
    """
    from tests.fixtures.registries import isolate
    from unified_kg_rag.adapters.renderers import base

    isolate(monkeypatch, base._RENDERERS)


@pytest.fixture
def skip_vertex_id_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the Neptune indexer's and retriever's vertex-id mode check.

    For tests that script every Gremlin request (or count them): the check
    reads each label once before its first use. It is pinned by
    ``tests/unit/test_neptune_vertex_ids.py``.
    """
    from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
    from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer

    async def configured(self: NeptuneRetriever, label: str) -> tuple[bool, None]:
        return self._neptune_config.vertex_ids == "custom", None

    monkeypatch.setattr(NeptuneIndexer, "_check_vertex_ids", lambda self, label: None)
    monkeypatch.setattr(NeptuneRetriever, "_vertex_lookup", configured)
    monkeypatch.setattr(NeptuneRetriever, "_settled", lambda self, label: False)


@pytest.fixture
def config() -> Config:
    """A default ``Config`` (all nested defaults, no external services)."""
    return Config()
