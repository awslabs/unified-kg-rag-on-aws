# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run's cache of the search-store lookups only its own writes change.

A pipeline run installs one (:func:`install_lookup_cache`) for its duration,
and the vector indexers of the run share it (several instances, worker
threads included: ``ContextThreadPoolExecutor`` copies the context). It
holds which index each alias points at and each index's mappings, read once
per run instead of before every read and write. While the run holds its
namespace's write lease no other run changes them, and every indexer
request that does (an alias swap or removal, an index deletion, a mapping
or ``_meta`` update) drops the entries it touches before it is sent, so a
failed or lost request leaves nothing stale. Outside a run (the direct
APIs) nothing is cached.

Entries are keyed by store (``store``: a value naming the store, so two
indexers of one store share their lookups) and name. A forget bumps a
generation: a read that started before it is not cached, as it may have
read the store before the change.

Kept in the shared kernel so the pipeline can install it without depending
on the adapters.
"""

from __future__ import annotations

import copy
import threading
from collections.abc import Callable, Collection, Hashable
from contextvars import ContextVar, Token
from typing import Any

__all__ = [
    "StoreLookupCache",
    "install_lookup_cache",
    "lookup_cache",
    "remove_lookup_cache",
]


class StoreLookupCache:
    """Alias targets and index mappings by (store, name); thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0
        self._aliases: dict[tuple[Hashable, str], str | None] = {}
        self._mappings: dict[tuple[Hashable, str], dict[str, Any]] = {}

    def alias(
        self, store: Hashable, alias: str, read: Callable[[], str | None]
    ) -> str | None:
        """The index ``alias`` points at (``None``: none), ``read`` once."""
        key = (store, alias)
        with self._lock:
            if key in self._aliases:
                return self._aliases[key]
            generation = self._generation
        value = read()
        with self._lock:
            if generation == self._generation:
                self._aliases[key] = value
        return value

    def mappings(
        self, store: Hashable, index: str, read: Callable[[], dict[str, Any]]
    ) -> dict[str, Any]:
        """``index``'s mappings, ``read`` once; a copy each time (callers
        may change what they get)."""
        key = (store, index)
        with self._lock:
            if key in self._mappings:
                return copy.deepcopy(self._mappings[key])
            generation = self._generation
        value = read()
        with self._lock:
            if generation == self._generation:
                self._mappings[key] = copy.deepcopy(value)
        return value

    def forget_aliases(self, store: Hashable) -> None:
        """Drop every alias of ``store`` (before a swap or removal: an
        alias may also name another one's index)."""
        with self._lock:
            self._generation += 1
            for key in [k for k in self._aliases if k[0] == store]:
                del self._aliases[key]

    def forget_mappings(self, store: Hashable, indices: Collection[str]) -> None:
        """Drop the mappings of ``indices`` (before they are updated, or the
        indices deleted)."""
        with self._lock:
            self._generation += 1
            for index in indices:
                self._mappings.pop((store, index), None)


_CACHE: ContextVar[StoreLookupCache | None] = ContextVar(
    "store_lookup_cache", default=None
)


def lookup_cache() -> StoreLookupCache | None:
    """This context's run cache, or ``None`` outside a run."""
    return _CACHE.get()


def install_lookup_cache() -> Token[StoreLookupCache | None]:
    """Give this context a new cache until :func:`remove_lookup_cache` is
    given the returned token."""
    return _CACHE.set(StoreLookupCache())


def remove_lookup_cache(token: Token[StoreLookupCache | None]) -> None:
    """Restore the cache before :func:`install_lookup_cache`."""
    _CACHE.reset(token)
