# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry of seed graph sources, keyed by ``processing.seed_graphs[].kind``.

A source turns one ``processing.seed_graphs`` item into a
:class:`~unified_kg_rag.domain.models.seed_graph.SeedGraph`: the nodes and
edges it states. Start-up loads every seed once
(``application.bootstrap.load_seed_graphs``) and converts it against the
run's resolved extraction schema into virtual documents
(``domain.ingestion.seed_graph.convert_seed``). A new format or service
registers its own kind, with no edit to the config model or the pipeline::

    @register_seed_source("my_format")
    def _load(context: SeedSourceContext) -> SeedGraph:
        options = context.options(MyOptions)
        return SeedGraph(nodes=read_nodes(context.source.path, options))

and is selected with ``kind: my_format``. The loader gets a
:class:`SeedSourceContext`: the seed's settings (``path``, ``options``,
``secret_arn`` ...), the run's config with the extraction schema already
resolved, and an AWS session built on first use. What it returns must
depend only on the bytes it read and the options, never on the order it
read them in (the conversion sorts). It raises ``SeedGraphError`` for a
seed it cannot read (:attr:`SeedSourceContext.terms` builds one naming the
seed); the CLIs exit 2 on it.

Every kind takes the options of :class:`SeedSourceOptions` (how the
conversion treats dangling edges and colliding names): a loader validates
``options`` with a subclass of it (:meth:`SeedSourceContext.options`).
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from unified_kg_rag.adapters.source_files.locations import SourceTerms, redact_location
from unified_kg_rag.domain.ingestion.seed_graph import SeedDangling
from unified_kg_rag.domain.models.config import Config, SeedGraphConfig
from unified_kg_rag.domain.models.seed_graph import SeedGraph
from unified_kg_rag.shared.exceptions import ConfigurationError, SeedGraphError

if TYPE_CHECKING:
    import boto3

# Where a seed's options sit in the config, for messages that name one.
OPTIONS_KEY = "processing.seed_graphs[].options"


@dataclass(frozen=True)
class SeedTerms(SourceTerms):
    """:class:`SourceTerms` of a seed source: errors are ``SeedGraphError``s
    starting with the seed's name (``seed graph 'procure': ...``)."""

    noun: str = "Seed source"
    options: str = OPTIONS_KEY
    purpose: str = "the seed graph"
    error: type[ConfigurationError] = SeedGraphError
    seed: str = ""

    def __call__(self, message: str) -> ConfigurationError:
        """``message`` as :attr:`error`, naming the seed."""
        return self.error(
            f"seed graph '{self.seed}': {message}" if self.seed else message
        )


# The terms of a seed source's errors, before a seed names them.
SEED_TERMS = SeedTerms()


class SeedSourceOptions(BaseModel):
    """The options every seed kind takes: how the conversion treats what the
    source states. A kind's options subclass it."""

    # An unknown option is a misspelling: it fails the import.
    model_config = ConfigDict(extra="forbid", frozen=True)

    dangling: SeedDangling = Field(
        default="error",
        description="An edge naming an unknown node: 'error' fails the "
        "import, 'skip' leaves the edge out (counted)",
    )
    qualifier_property: str | None = Field(
        default=None,
        description="Node property whose value on_name_collision: qualify "
        "appends to a colliding name (else the IRI's local name, else the id)",
    )


# The option names every kind takes.
COMMON_OPTIONS = frozenset(SeedSourceOptions.model_fields)

OptionsT = TypeVar("OptionsT", bound=SeedSourceOptions)


def _default_session(config: Config) -> boto3.Session:
    import boto3

    return boto3.Session(
        profile_name=config.aws.profile_name, region_name=config.aws.region_name
    )


class SeedSourceContext:
    """What a seed loader is given; a
    :class:`.source_files.locations.SourceClientProvider`, so the shared
    location readers read its S3 objects.

    Attributes:
        source: The ``processing.seed_graphs`` item (``name``, ``kind``,
            ``path``, ``options``, ``secret_arn``, the caps). Never log
            ``path`` as is (:attr:`location` is the printable form) or the
            secret.
        config: The run's config, its extraction schema already in the
            inline keys (``config.processing.graph_extraction``).
        terms: How the source's errors name it (``SeedGraphError``s
            starting with the seed's name).
    """

    def __init__(
        self,
        source: SeedGraphConfig,
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
        self.terms = SeedTerms(kind=source.kind, seed=source.name)
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

    def options(self, model: type[OptionsT]) -> OptionsT:
        """``source.options`` validated as ``model``.

        Raises:
            SeedGraphError: An option is unknown or invalid.
        """
        return seed_options(self.source, model)


def seed_options(source: SeedGraphConfig, model: type[OptionsT]) -> OptionsT:
    """``source.options`` validated as ``model``.

    Raises:
        SeedGraphError: An option is unknown or invalid, each named.
    """
    try:
        return model.model_validate(source.options)
    except ValidationError as e:
        raise SeedGraphError(
            f"seed graph '{source.name}': "
            + "; ".join(
                "options"
                + "".join(f".{part}" for part in detail["loc"])
                + f": {detail['msg']}"
                for detail in e.errors(include_url=False, include_input=False)
            )
        ) from e


SeedLoader = Callable[[SeedSourceContext], SeedGraph]
LoaderT = TypeVar("LoaderT", bound=SeedLoader)

_REGISTRY: dict[str, SeedLoader] = {}
# Kind -> (module, how to install it) of the optional dependency its loader
# imports.
_REQUIRES: dict[str, tuple[str, str]] = {}
# The form processing.seed_graphs[].kind takes (SeedGraphConfig.kind).
_KIND = re.compile(r"^[a-z][a-z0-9_]*$")


def register_seed_source(
    kind: str, *, requires: tuple[str, str] | None = None
) -> Callable[[LoaderT], LoaderT]:
    """Register the decorated loader under ``kind``.

    The loader is called as ``loader(context)`` with a
    :class:`SeedSourceContext` and returns a ``SeedGraph`` that depends only
    on what it read and the options; it raises ``SeedGraphError`` for a seed
    it cannot read. ``requires`` is the ``(module, how to install it)`` of
    the optional dependency the loader imports, or None: the CLIs check it
    is installed before any model call. A later registration of the same
    kind replaces the earlier one, requirement included, so a deployment can
    override a built-in.

    Raises:
        ValueError: ``kind`` is not of the form ``seed_graphs[].kind`` takes.
    """
    if not _KIND.match(kind):
        raise ValueError(
            f"Cannot register a seed source as {kind!r}: a kind is "
            "lower-case letters, digits and '_', starting with a letter"
        )

    def decorator(loader: LoaderT) -> LoaderT:
        _REGISTRY[kind] = loader
        if requires is None:
            _REQUIRES.pop(kind, None)
        else:
            _REQUIRES[kind] = requires
        return loader

    return decorator


def get_seed_source(kind: str) -> SeedLoader:
    """The loader registered under ``kind``.

    Raises:
        SeedGraphError: Nothing is registered under ``kind``.
    """
    loader = _REGISTRY.get(kind)
    if loader is None:
        raise SeedGraphError(
            f"Unknown processing.seed_graphs[].kind '{kind}'; registered: "
            f"{', '.join(registered_seed_sources())}"
        )
    return loader


def seed_source_requirement(kind: str) -> tuple[str, str] | None:
    """The ``(module, how to install it)`` ``kind`` declared, or None."""
    return _REQUIRES.get(kind)


def registered_seed_sources() -> list[str]:
    """The registered kinds, sorted."""
    return sorted(_REGISTRY)
