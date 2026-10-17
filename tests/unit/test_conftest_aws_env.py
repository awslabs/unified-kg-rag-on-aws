# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The suite's environment isolation leaves ``aws``-marked tests alone.

``_isolated_aws_env`` clears the store endpoints and the container
credential variables and sets fake keys for every test; an ``aws`` test
(real stores, ``-m aws``) would then always skip for lack of endpoints, or
call AWS with the fake keys. Runs an inner session with the suite's
fixtures over dummy values; nothing calls AWS.
"""

from __future__ import annotations

import pytest

pytest_plugins = ("pytester",)

pytestmark = pytest.mark.unit

_INNER_TESTS = """
import os

import pytest


@pytest.mark.aws
def test_aws_marked_sees_the_real_environment():
    assert os.environ["UKR__AWS__NEPTUNE__ENDPOINT"] == "neptune.example.invalid"
    assert os.environ["AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"] == "/dummy-creds"
    assert "AWS_ACCESS_KEY_ID" not in os.environ


def test_unmarked_is_isolated():
    assert "UKR__AWS__NEPTUNE__ENDPOINT" not in os.environ
    assert "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI" not in os.environ
    assert os.environ["AWS_ACCESS_KEY_ID"] == "testing"
"""


def test_aws_marked_tests_keep_the_real_environment(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytester.makeini(
        "[pytest]\nasyncio_default_fixture_loop_scope = function\n"
        "markers =\n    aws: real AWS\n"
    )
    pytester.makeconftest(
        "from tests.conftest import _block_real_aws_http, _isolated_aws_env\n"
    )
    pytester.makepyfile(_INNER_TESTS)
    monkeypatch.setenv("UKR__AWS__NEPTUNE__ENDPOINT", "neptune.example.invalid")
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/dummy-creds")
    monkeypatch.delenv("AWS_ACCESS_KEY_ID")

    result = pytester.runpytest("-p", "no:cacheprovider", "--strict-markers")

    result.assert_outcomes(passed=2)


def test_plain_pytest_run_deselects_aws_tests(pytestconfig: pytest.Config) -> None:
    # aws tests keep the real environment above, so a bare `pytest` with the
    # store endpoints exported must not select them: the ini addopts carry
    # the selector and an explicit `-m aws` on the command line replaces it.
    addopts = pytestconfig.getini("addopts")
    assert "-m" in addopts
    assert addopts[addopts.index("-m") + 1] == "not aws"
