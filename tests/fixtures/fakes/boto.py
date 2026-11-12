# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A boto3 ``Session`` stand-in for the Bedrock factories.

The factories only need a session to hand to their clients; capability and
model-resolution logic never calls the returned client, so ``client()`` returns
an opaque client (its only surface: the event handlers registered on it) and
records which services were asked for.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


class FakeEvents:
    """``client.meta.events``: records the handlers registered on it."""

    def __init__(self) -> None:
        self.handlers: list[tuple[str, Any]] = []

    def register(self, event_name: str, handler: Any) -> None:
        self.handlers.append((event_name, handler))


class FakeClient:
    def __init__(self) -> None:
        self.meta = SimpleNamespace(events=FakeEvents())


class FakeBotoSession:
    profile_name = "default"

    def __init__(self) -> None:
        self.clients_requested: list[str] = []

    def client(self, service_name: str, **kwargs: Any) -> Any:
        self.clients_requested.append(service_name)
        return FakeClient()

    def get_credentials(self) -> Any:
        return None  # no assume-role short-circuit path
