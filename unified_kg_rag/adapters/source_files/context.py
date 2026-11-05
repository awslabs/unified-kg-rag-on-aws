# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What every source registry shares: the context a loader is given and the
kind -> loader registry (a source names invalid settings with
:func:`unified_kg_rag.shared.utils.validation.format_validation_errors`).

The schema-source registry (:mod:`.schema_sources.registry`) and the
seed-source registry (:mod:`.seed_sources.registry`) build on these and keep
their own public names, docstrings and messages.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from unified_kg_rag.domain.models.config import Config

from .locations import redact_location

if TYPE_CHECKING:
    import boto3

# The form a source kind takes (SchemaSourceConfig.kind, SeedGraphConfig.kind).
KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class _Located(Protocol):
    @property
    def path(self) -> str | None: ...


SourceT = TypeVar("SourceT", bound=_Located)
LoaderT = TypeVar("LoaderT", bound=Callable[..., Any])


def _default_session(config: Config) -> boto3.Session:
    import boto3

    return boto3.Session(
        profile_name=config.aws.profile_name, region_name=config.aws.region_name
    )


class SourceContextBase(Generic[SourceT]):
    """What a loader of any source kind is given: its settings, the run's
    config and the run's AWS session, built on first use; a
    :class:`.locations.SourceClientProvider`, so the shared location readers
    read its S3 objects.

    Attributes:
        source: The source's settings. Never log ``path`` as is
            (:attr:`location` is the printable form) or a secret.
        config: The run's config.
    """

    def __init__(
        self,
        source: SourceT,
        config: Config,
        *,
        session_factory: Callable[[], boto3.Session] | None = None,
    ) -> None:
        """``session_factory`` builds the AWS session, once, on the first
        :meth:`boto_session` (by default from ``aws.profile_name`` and
        ``aws.region_name``), so a loader that makes no AWS call builds
        none."""
        self.source = source
        self.config = config
        self._session_factory = session_factory or (lambda: _default_session(config))
        self._session: boto3.Session | None = None
        self._lock = threading.Lock()

    @property
    def location(self) -> str:
        """``source.path`` as logs and errors may show it
        (:func:`redact_location`)."""
        return redact_location(self.source.path)

    def boto_session(self) -> boto3.Session:
        """The run's AWS session, built on the first call."""
        with self._lock:
            if self._session is None:
                self._session = self._session_factory()
            return self._session

    def client(self, service_name: str, **kwargs: Any) -> Any:
        """A client of the run's session in the region the config sends
        ``service_name`` to: ``aws.bedrock_region`` for the Bedrock
        services, ``aws.region_name`` for every other one."""
        region = (
            self.config.aws.bedrock_region
            if service_name.startswith("bedrock")
            else self.config.aws.region_name
        )
        kwargs.setdefault("region_name", region)
        # Any: the service name is only known at run time.
        session: Any = self.boto_session()
        return session.client(service_name, **kwargs)


class KindRegistry(Generic[LoaderT]):
    """Loaders by kind, and the optional dependency each declared.

    Attributes:
        loaders: Kind -> loader.
        requires: Kind -> ``(module, how to install it)`` of the optional
            dependency its loader imports.
    """

    def __init__(self) -> None:
        self.loaders: dict[str, LoaderT] = {}
        self.requires: dict[str, tuple[str, str]] = {}

    def register(
        self, kind: str, loader: LoaderT, requires: tuple[str, str] | None
    ) -> None:
        """Register ``loader`` under ``kind``, replacing an earlier one and
        its requirement."""
        self.loaders[kind] = loader
        if requires is None:
            self.requires.pop(kind, None)
        else:
            self.requires[kind] = requires

    def get(self, kind: str) -> LoaderT | None:
        """The loader registered under ``kind``, or None."""
        return self.loaders.get(kind)

    def requirement(self, kind: str) -> tuple[str, str] | None:
        """The ``(module, how to install it)`` ``kind`` declared, or None."""
        return self.requires.get(kind)

    def kinds(self) -> list[str]:
        """The registered kinds, sorted."""
        return sorted(self.loaders)
