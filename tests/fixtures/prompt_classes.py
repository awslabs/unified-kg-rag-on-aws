# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The prompt classes the package ships (``unified_kg_rag.domain.prompts``)."""

from __future__ import annotations

import importlib
import inspect
import pkgutil

import unified_kg_rag.domain.prompts as prompts
from unified_kg_rag.domain.prompts.base import BasePrompt


def shipped_prompts() -> list[type[BasePrompt]]:
    """Every ``BasePrompt`` subclass defined in the prompts package, by name."""
    found: set[type[BasePrompt]] = set()
    for info in pkgutil.iter_modules(prompts.__path__):
        module = importlib.import_module(f"{prompts.__name__}.{info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if issubclass(cls, BasePrompt) and cls is not BasePrompt:
                found.add(cls)
    return sorted(found, key=lambda cls: cls.__name__)
