# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Names a CLI module imports on first use, not when it loads.

The chain and the pipeline pull in LangChain, the strategies and the store
SDKs (seconds), which ``--help`` and a refused command line never need. A CLI
module declares those names here; they stay attributes of the module (a
test patches ``run_rag_chain.GraphRAGChain`` as before), and its code binds
them before using them.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any


def lazy_names(
    namespace: dict[str, Any], names: Mapping[str, Callable[[], ModuleType]]
) -> tuple[Callable[[str], Any], Callable[[], None]]:
    """The module ``__getattr__`` and the loader of ``names`` (name -> a
    function that imports, statically, and returns the module defining it)
    for the module whose globals are ``namespace``.

    The ``__getattr__`` imports a name on its first attribute access and
    binds it in ``namespace``; the loader binds every name not bound yet (a
    test's patch, bound already, is kept), for code reading them as
    globals.
    """

    def getattr_(name: str) -> Any:
        if name not in names:
            raise AttributeError(
                f"module {namespace['__name__']!r} has no attribute {name!r}"
            )
        value = getattr(names[name](), name)
        namespace[name] = value
        return value

    def load() -> None:
        for name in names:
            if name not in namespace:
                getattr_(name)

    return getattr_, load
