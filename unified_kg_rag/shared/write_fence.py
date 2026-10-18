# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The write fence of a run: stop writing once the run lost the right to.

A pipeline run with the doc-status registry holds its index namespace's write
lease (``application.ingestion.namespace_lease``) and installs it as the
fence of its context; worker threads inherit it through
``ContextThreadPoolExecutor``. The registry writes of the incremental path,
the indexing manager's phases and the store adapters' write requests (a
Neptune batch or drop, an OpenSearch bulk request) call
:func:`check_write_fence` first, so a run whose lease was lost stops at its
next request instead of writing over the run that holds it now. A pipeline
run also installs a fence that ends its worker threads' writes once a stop
was requested (``request_stop``). Fences stack: the check asks each one, the
innermost first. Outside a fenced run (the direct APIs) it does nothing.

Kept in the shared kernel so the adapters can check it without depending on
the application layer.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Protocol

__all__ = [
    "WriteFence",
    "check_write_fence",
    "install_write_fence",
    "remove_write_fence",
]


class WriteFence(Protocol):
    """Anything that can tell whether the run may still write."""

    def check(self) -> None:
        """Raise when the run must not write any more."""
        ...


_FENCES: ContextVar[tuple[WriteFence, ...]] = ContextVar("write_fences", default=())


def check_write_fence() -> None:
    """Raise if this context's run lost the right to write (no-op outside a
    fenced run). A lease fence raises ``NamespaceLeaseError``, a requested
    stop ``PipelineInterrupted``."""
    for fence in reversed(_FENCES.get()):
        fence.check()


def install_write_fence(fence: WriteFence) -> Token[tuple[WriteFence, ...]]:
    """Add ``fence`` to this context's fences until :func:`remove_write_fence`
    is given the returned token."""
    return _FENCES.set((*_FENCES.get(), fence))


def remove_write_fence(token: Token[tuple[WriteFence, ...]]) -> None:
    """Restore the fences before :func:`install_write_fence` added one."""
    _FENCES.reset(token)
