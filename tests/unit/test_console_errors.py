# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console API's error contract: every error of the package answers
with the class every surface names it by (``error_class``) and the HTTP
status of that class, a request the console refuses is ``invalid_argument``
and an unreachable store a 503 ``store_unreachable``, not a bare 500."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from tests.fixtures.console_demo import requires_console
from unified_kg_rag.shared.exceptions import (
    ConfigurationError,
    ErrorClass,
    IndexNotFoundError,
    InvalidArgumentError,
    InvalidFilterError,
    MethodologyNotIndexedError,
    ModelUnreachableError,
    PipelineExecutionError,
    QueryTimeoutError,
    StoreConfigurationError,
    StoreConnectionError,
)

pytestmark = [pytest.mark.unit, requires_console]

HEADERS = {"X-Console-Request": "1"}
REPOSITORY = Path(__file__).resolve().parents[2]


def _client(tmp_path: Path, config_path: Path | None = None) -> Any:
    from fastapi.testclient import TestClient

    from unified_kg_rag.application.cli.console.server import ConsoleOptions, create_app

    options = ConsoleOptions(
        workspace=tmp_path,
        cache_directory=tmp_path / "cache",
        config_path=config_path,
    )
    return TestClient(
        create_app(options),
        base_url="http://127.0.0.1",
        raise_server_exceptions=False,
    )


@pytest.mark.parametrize(
    ("error", "status", "code", "error_class"),
    [
        (InvalidFilterError("Filter 'x' has no value"), 400, "invalid_filter", None),
        (InvalidArgumentError("choose a strategy"), 400, "invalid", "invalid_argument"),
        (MethodologyNotIndexedError("holds graphrag only"), 404, "not_indexed", None),
        (IndexNotFoundError("No indices found"), 404, "index_not_found", None),
        (ConfigurationError("no region"), 400, "configuration", None),
        (StoreConfigurationError("mapping refused"), 400, "configuration", None),
        (QueryTimeoutError("past", timeout_seconds=1.0), 504, "timeout", None),
        (StoreConnectionError("no route"), 503, "store_unreachable", None),
        (ModelUnreachableError("refused"), 503, "model_unreachable", None),
        (PipelineExecutionError("broke"), 500, "failure", None),
        (RuntimeError("unexpected"), 500, "failure", None),
    ],
    ids=lambda value: type(value).__name__ if isinstance(value, Exception) else None,
)
def test_an_error_answers_with_its_class_and_the_class_s_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status: int,
    code: str,
    error_class: str | None,
) -> None:
    # Regression (rv25): only the console's own refusals, ConfigurationError
    # and DataProcessingError had a handler: a StoreConnectionError was a
    # plain-text 500, and a plain ConfigurationError "invalid_argument".
    client = _client(tmp_path)

    async def raises(*_: Any, **__: Any) -> Any:
        raise error

    monkeypatch.setattr(client.app.state.console.ask, "held_methodologies", raises)
    response = client.get("/api/namespace/methodologies")
    assert response.status_code == status, response.text
    body = response.json()
    assert body["code"] == code
    assert body["error_class"] == (error_class or code)
    assert body["error"] == str(error)


def test_every_error_class_has_a_status() -> None:
    from unified_kg_rag.application.cli.console.server import HTTP_STATUS

    assert set(HTTP_STATUS) == set(ErrorClass)


def test_the_console_s_own_refusals_are_invalid_arguments(tmp_path: Path) -> None:
    # A request value the console refuses is the caller's to change, as
    # every surface names it: not "configuration".
    client = _client(tmp_path)
    for refused in (
        client.get("/api/strategies", params={"suffix": "../x"}),
        client.post(
            "/api/jobs/evaluate",
            json={"dataset": "workspace:none.json", "strategies": []},
            headers=HEADERS,
        ),
        client.get("/api/graphs/data", params={"ref": "nowhere:x"}),
    ):
        assert 400 <= refused.status_code < 500, refused.text
        assert refused.json()["error_class"] == "invalid_argument", refused.text


@pytest.mark.integration
def test_an_unreachable_store_is_a_503_store_unreachable(tmp_path: Path) -> None:
    # The architecture probe: the stores on a closed port answered a
    # plain-text "Internal Server Error".
    data = yaml.safe_load((REPOSITORY / "docker" / "config.local.yaml").read_text())
    data["providers"]["doc_status"] = {"enabled": False}
    data["aws"] = {"opensearch": {"port": 1}, "neptune": {"port": 1}}
    config = tmp_path / "down.yaml"
    config.write_text(yaml.safe_dump(data))
    response = _client(tmp_path, config).get("/api/namespace/methodologies")
    assert response.status_code == 503, response.text
    assert response.json()["error_class"] == "store_unreachable"
