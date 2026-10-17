# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run's cache of ingestion model responses (a LangChain ``BaseCache``).

A re-run with the same pipeline id resumes after the last finished stage; this
cache also spares the model calls the interrupted stage had already made.
``CachingLLMFactory`` attaches it to every model an LLM factory builds for
``ModelPurpose.INGESTION``, so it applies to any backend and any stage that
calls a model, without knowing their prompts.

Entries live in immutable JSON segment files in the run's cache directory
(synced to S3 with the stage cache). Only entries of earlier runs are served,
each at most once per run: a retry within the run (after a response failed to
parse, say) calls the model again, and its response replaces the old one. A
response cut off at the output-token limit is not stored, since its failure is
not retried.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.messages import message_to_dict, messages_from_dict
from langchain_core.outputs import ChatGeneration, Generation

from unified_kg_rag.adapters.llm.chain_factory import _truncation_reason
from unified_kg_rag.domain.models import ModelPurpose
from unified_kg_rag.ports.model_factory import LLMFactoryPort
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

# Object reprs in a non-serializable model's llm_string differ per process.
_OBJECT_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")
_SEGMENT_VERSION = 1


def _cache_key(prompt: str, llm_string: str) -> str:
    material = f"{prompt}\n{_OBJECT_ADDRESS.sub('', llm_string)}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _dump_generation(generation: Generation) -> dict[str, Any]:
    if isinstance(generation, ChatGeneration):
        return {
            "message": message_to_dict(generation.message),
            "generation_info": generation.generation_info,
        }
    return {"text": generation.text, "generation_info": generation.generation_info}


def _load_generation(data: dict[str, Any]) -> Generation:
    if "message" in data:
        (message,) = messages_from_dict([data["message"]])
        return ChatGeneration(
            message=message, generation_info=data.get("generation_info")
        )
    return Generation(text=data["text"], generation_info=data.get("generation_info"))


class LLMResponseCache(BaseCache):
    """Model responses of one run, kept in ``directory`` (see the module doc).

    Inactive (every lookup a miss, nothing stored) until ``open``; the
    pipeline opens it once the run's cache directory is in place.
    """

    def __init__(self, flush_every: int = 100) -> None:
        self.flush_every = flush_every
        # Re-entrant: a stop requested during a flush flushes again (no-op).
        self._lock = threading.RLock()
        self._directory: Path | None = None
        self._lookups = True
        self._entries: dict[str, list[dict[str, Any]]] = {}
        self._served: set[str] = set()
        self._pending: list[dict[str, Any]] = []
        self._on_segment: Callable[[Path], None] | None = None
        self.hits = 0
        self.misses = 0

    def open(
        self,
        directory: Path,
        *,
        lookups: bool = True,
        on_segment: Callable[[Path], None] | None = None,
    ) -> None:
        """Load the segments in ``directory`` and store new ones there.

        Args:
            lookups: ``False`` stores responses but serves none (a force
                rebuild).
            on_segment: Called with each segment file once it is written
                (the pipeline uploads it to S3).
        """
        entries: dict[str, list[dict[str, Any]]] = {}
        # Named by write time, so a later response for a key replaces an earlier one.
        for path in sorted(directory.glob("*.json")) if directory.is_dir() else []:
            try:
                segment = json.loads(path.read_text(encoding="utf-8"))
                for entry in segment["entries"]:
                    entries[entry["key"]] = entry["generations"]
            except (OSError, ValueError, KeyError, TypeError) as e:
                logger.warning(
                    "Skipping unreadable LLM cache segment '%s': %s", path, e
                )
        with self._lock:
            self._directory = directory
            self._lookups = lookups
            self._entries = entries
            self._served = set()
            self._pending = []
            self._on_segment = on_segment
            self.hits = self.misses = 0
        logger.info(
            "LLM response cache at '%s': %s cached responses%s",
            directory,
            len(entries),
            "" if lookups else " (not served: force rebuild)",
        )

    def close(self) -> None:
        """Write the pending responses and deactivate the cache."""
        self.flush()
        with self._lock:
            if self._directory is not None:
                logger.info(
                    "LLM response cache: %s hits, %s misses", self.hits, self.misses
                )
            self._directory = None
            self._entries = {}

    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        key = _cache_key(prompt, llm_string)
        with self._lock:
            if self._directory is None:
                return None
            cached = self._entries.get(key) if self._lookups else None
            if cached is None or key in self._served:
                self.misses += 1
                return None
            self._served.add(key)
            self.hits += 1
        return [_load_generation(g) for g in cached]

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        if any(
            isinstance(g, ChatGeneration) and _truncation_reason(g.message)
            for g in return_val
        ):
            return
        entry = {
            "key": _cache_key(prompt, llm_string),
            "generations": [_dump_generation(g) for g in return_val],
        }
        with self._lock:
            if self._directory is None:
                return
            self._pending.append(entry)
            full = len(self._pending) >= self.flush_every
        if full:
            self.flush()

    def flush(self) -> None:
        """Write the pending responses as a new segment (no-op when none)."""
        with self._lock:
            directory, pending = self._directory, self._pending
            if directory is None or not pending:
                return
            self._pending = []
            on_segment = self._on_segment
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{time.time_ns():020d}-{secrets.token_hex(4)}.json"
            temp = path.with_suffix(".tmp")
            temp.write_text(
                json.dumps(
                    {"version": _SEGMENT_VERSION, "entries": pending}, default=str
                ),
                encoding="utf-8",
            )
            temp.replace(path)
        if on_segment is not None:
            on_segment(path)

    def clear(self, **kwargs: Any) -> None:
        with self._lock:
            self._entries = {}
            self._served = set()
            self._pending = []


class CachingLLMFactory:
    """An ``LLMFactoryPort`` whose ingestion models use ``cache``.

    Models built for any other purpose (queries, evaluation) are returned as
    the wrapped factory built them. Other attributes (``is_transient_error``,
    ``transient_retry``) are the wrapped factory's.
    """

    def __init__(self, factory: LLMFactoryPort, cache: BaseCache) -> None:
        self.factory = factory
        self.cache = cache

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        model = self.factory.get_model(model_id, **kwargs)
        purpose = ModelPurpose(kwargs.get("model_purpose", ModelPurpose.QUERY))
        if purpose is not ModelPurpose.INGESTION or not hasattr(model, "cache"):
            return model
        # A copy: a factory may hand the same instance to other callers.
        return model.model_copy(update={"cache": self.cache})

    def get_model_info(self, model_id: Any) -> Any:
        return self.factory.get_model_info(model_id)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.factory, name)
