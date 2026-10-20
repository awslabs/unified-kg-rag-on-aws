# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The write fence of a run: stop writing once the run lost the right to.

A pipeline run installs a fence that ends its worker threads' writes once a
stop was requested (``request_stop``, or Ctrl-C); with the doc-status
registry it also holds its index namespace's write lease
(``application.ingestion.namespace_lease``) and installs it as a second
fence, which fails once the lease was lost, is about to expire or was
released at the end of the run. Worker threads inherit the fences through
``ContextThreadPoolExecutor``. The registry writes of the incremental path,
the indexing manager's phases and the store adapters' requests call
:func:`check_write_fence` first: a Neptune write batch, edge write or drop,
and every attempt of an OpenSearch indexing-client request but a GET or HEAD
(the bulk writes, index and alias changes, and also the POST searches and
reads of the indexing path). A run that may not write any more then stops at
its next request instead of writing over the run that holds the lease now.
Fences stack: the check asks each one, the innermost (the lease) first.
Outside a fenced run (the direct APIs) it does nothing.

Kept in the shared kernel so the adapters can check it without depending on
the application layer.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Protocol

__all__ = [
    "INDEXING_REQUEST_TIMEOUT_SECONDS",
    "INGESTION_REQUEST_TIMEOUT_SECONDS",
    "REGISTRY_CONNECT_TIMEOUT_SECONDS",
    "REGISTRY_READ_TIMEOUT_SECONDS",
    "WriteFence",
    "check_fences_held",
    "check_write_fence",
    "install_write_fence",
    "remove_write_fence",
]

# The longest single store requests of the indexing path, which a lease must
# outlast with margin (indexing.lease_seconds at least three times the
# longer one, see namespace_lease).
#
# One Neptune ingestion request (a write, a read-back, a drop of a batch).
# Above Neptune's own default query timeout (neptune_query_timeout, 120 s),
# so the server ends a long but healthy request first, and the OpenSearch
# indexing client's per-request timeout; a server that does not answer at
# all fails the request instead of blocking ingestion.
INGESTION_REQUEST_TIMEOUT_SECONDS = 180.0
# The per-request timeout of the OpenSearch indexing (sync) client.
INDEXING_REQUEST_TIMEOUT_SECONDS = 180.0
# The bounds of one attempt of a DynamoDB registry request (botocore's
# defaults are 60 s each). Every attempt of a registry write checks the fence
# first, so one attempt is the most a fenced write can be late by.
REGISTRY_CONNECT_TIMEOUT_SECONDS = 10.0
REGISTRY_READ_TIMEOUT_SECONDS = 60.0


class WriteFence(Protocol):
    """Anything that can tell whether the run may still write."""

    def check(self) -> None:
        """Raise when the run must not write any more (before a write)."""
        ...

    def check_held(self) -> None:
        """Raise when the run must stop, at a point that writes nothing:
        :meth:`check` without anything that holds the run's first write
        back (a fence without it is asked :meth:`check`)."""
        ...


_FENCES: ContextVar[tuple[WriteFence, ...]] = ContextVar("write_fences", default=())


def check_write_fence() -> None:
    """Raise if this context's run lost the right to write (no-op outside a
    fenced run). The lease fence raises ``NamespaceLeaseError``; the stop
    fence raises ``PipelineInterrupted`` in any thread but the one running
    the pipeline, which unwinds the stop itself."""
    for fence in reversed(_FENCES.get()):
        fence.check()


def check_fences_held() -> None:
    """:func:`check_write_fence` at a point that writes nothing (a stage
    start): a fence that holds a run's first write back (the lease's
    takeover wait) does not wait here (its ``check_held``)."""
    for fence in reversed(_FENCES.get()):
        getattr(fence, "check_held", fence.check)()


def install_write_fence(fence: WriteFence) -> Token[tuple[WriteFence, ...]]:
    """Add ``fence`` to this context's fences until :func:`remove_write_fence`
    is given the returned token."""
    return _FENCES.set((*_FENCES.get(), fence))


def remove_write_fence(token: Token[tuple[WriteFence, ...]]) -> None:
    """Restore the fences before :func:`install_write_fence` added one."""
    _FENCES.reset(token)
