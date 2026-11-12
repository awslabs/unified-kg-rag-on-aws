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

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from unified_kg_rag.adapters.source_files.context import SourceContextBase
from unified_kg_rag.adapters.source_files.locations import SourceTerms
from unified_kg_rag.domain.ingestion.seed_graph import SeedDangling
from unified_kg_rag.domain.models.config import Config, SeedGraphConfig
from unified_kg_rag.domain.models.seed_graph import SeedGraph
from unified_kg_rag.shared.exceptions import ConfigurationError, SeedGraphError
from unified_kg_rag.shared.registry import Registry
from unified_kg_rag.shared.utils.validation import format_validation_errors

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
    # A seed's size cap is one of its settings, not an option of its kind.
    max_bytes: str = "processing.seed_graphs[].max_bytes"
    purpose: str = "the seed graph"
    error: type[ConfigurationError] = SeedGraphError
    seed: str = ""

    def __call__(self, message: str) -> ConfigurationError:
        """``message`` as :attr:`error`, naming the seed."""
        return self.error(
            f"seed graph '{self.seed}': {message}" if self.seed else message
        )


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


class SeedSourceContext(SourceContextBase[SeedGraphConfig]):
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
        super().__init__(source, config, session_factory=session_factory)
        self.terms = SeedTerms(kind=source.kind, seed=source.name)

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
            f"seed graph '{source.name}': {format_validation_errors('options', e)}"
        ) from e


def statement_problems(error: ValidationError) -> str:
    """What makes a node or edge a source read invalid, each field named
    (``name: String should have at most 1000 characters``)."""
    return format_validation_errors("", error, root=None)


SeedLoader = Callable[[SeedSourceContext], SeedGraph]
LoaderT = TypeVar("LoaderT", bound=SeedLoader)

_KINDS: Registry[str, SeedLoader] = Registry("seed source")
_REGISTRY = _KINDS.entries
# Kind -> (module, how to install it) of the optional dependency its loader
# imports.
_REQUIRES = _KINDS.requires


def register_seed_source(
    kind: str, *, requires: tuple[str, str] | None = None
) -> Callable[[LoaderT], LoaderT]:
    """Register the decorated loader under ``kind``.

    The loader is called as ``loader(context)`` with a
    :class:`SeedSourceContext` and returns a ``SeedGraph`` that depends only
    on what it read and the options; it raises ``SeedGraphError`` for a seed
    it cannot read. ``requires`` is the ``(module, how to install it)`` of
    the optional dependency the loader imports, or None: the CLIs check it
    is installed before any model call. The name follows the one registry
    rule (``shared.registry``): case-insensitive, a later registration
    replaces.

    Raises:
        ValueError: ``kind`` is not of the form ``seed_graphs[].kind`` takes.
    """
    _KINDS.check_name(kind)

    def decorator(loader: LoaderT) -> LoaderT:
        _KINDS.register(kind, loader, requires=requires)
        return loader

    return decorator


def get_seed_source(kind: str) -> SeedLoader:
    """The loader registered under ``kind``.

    Raises:
        SeedGraphError: Nothing is registered under ``kind``.
    """
    loader = _KINDS.get(kind)
    if loader is None:
        raise SeedGraphError(
            f"Unknown processing.seed_graphs[].kind '{kind}'; registered: "
            f"{', '.join(registered_seed_sources())}"
        )
    return loader


def seed_source_requirement(kind: str) -> tuple[str, str] | None:
    """The ``(module, how to install it)`` ``kind`` declared, or None."""
    return _KINDS.requirement(kind)


def registered_seed_sources() -> list[str]:
    """The registered kinds, sorted."""
    return _KINDS.names()
