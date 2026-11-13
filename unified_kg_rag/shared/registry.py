# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The one registry rule of every extension seam keyed by a name.

Providers, renderers, schema and seed sources, analyzers and file loaders
each keep a :class:`Registry`, so a name means the same everywhere:

- A name is matched case-insensitively and without surrounding space
  (normalized to lower case), and must then match the registry's pattern
  (by default lower-case letters, digits and ``_``, starting with a letter)
  and not be one of its reserved names; anything else is refused with
  ``ValueError`` at registration.
- A later registration under the same name replaces the earlier one, its
  requirement included, so a deployment can override a built-in.
- ``requires=(module, how to install it)`` declares the optional
  dependency an entry imports; :meth:`Registry.missing` names it when it is
  not installed, so a caller fails before the work starts.

Search strategies are keyed by ``SearchStrategy`` and refuse a second
class under one name (``adapters.retrieval.strategy_registry``).
"""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Collection, Iterator
from typing import Generic, TypeVar

NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")

V = TypeVar("V")


def module_installed(module: str) -> bool:
    """Whether ``module`` can be imported (without importing it)."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


class Registry(Generic[V]):
    """Entries by name under the module's rule.

    Attributes:
        what: What an entry is, for messages ("schema source").
        entries: Normalized name -> entry.
        requires: Normalized name -> ``(module, how to install it)``.
    """

    def __init__(
        self,
        what: str,
        *,
        pattern: re.Pattern[str] = NAME_PATTERN,
        reserved: Collection[str] = (),
    ) -> None:
        self.what = what
        self.pattern = pattern
        self.reserved = frozenset(reserved)
        self.entries: dict[str, V] = {}
        self.requires: dict[str, tuple[str, str]] = {}

    @staticmethod
    def normalize(name: str) -> str:
        return name.strip().lower()

    def register(
        self, name: str, entry: V, *, requires: tuple[str, str] | None = None
    ) -> str:
        """Register ``entry`` under ``name`` (replacing an earlier one and
        its requirement); return the normalized name.

        Raises:
            ValueError: The normalized name does not match the pattern or
                is reserved.
        """
        key = self.check_name(name)
        self.entries[key] = entry
        if requires is None:
            self.requires.pop(key, None)
        else:
            self.requires[key] = requires
        return key

    def unregister(self, name: str) -> None:
        """Remove ``name``'s entry and requirement, if registered."""
        key = self.normalize(name)
        self.entries.pop(key, None)
        self.requires.pop(key, None)

    def check_name(self, name: str) -> str:
        """``name`` normalized, if it may be registered.

        Raises:
            ValueError: It does not match the pattern or is reserved.
        """
        key = self.normalize(name)
        if key in self.reserved or not self.pattern.fullmatch(key):
            reserved = (
                f", and is not {', '.join(sorted(self.reserved))}"
                if self.reserved
                else ""
            )
            raise ValueError(
                f"Cannot register a {self.what} as {name!r}: a name (any case) "
                f"matches {self.pattern.pattern}{reserved}"
            )
        return key

    def get(self, name: str) -> V | None:
        """The entry registered under ``name``, or None."""
        return self.entries.get(self.normalize(name))

    def requirement(self, name: str) -> tuple[str, str] | None:
        """The ``(module, how to install it)`` ``name`` declared, or None."""
        return self.requires.get(self.normalize(name))

    def missing(self, name: str) -> tuple[str, str] | None:
        """``name``'s declared requirement when it is not installed."""
        requirement = self.requirement(name)
        if requirement is None or module_installed(requirement[0]):
            return None
        return requirement

    def names(self) -> list[str]:
        """The registered names, sorted."""
        return sorted(self.entries)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.normalize(name) in self.entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)
