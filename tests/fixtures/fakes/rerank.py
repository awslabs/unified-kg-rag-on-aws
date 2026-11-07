# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A fake ``sentence_transformers.CrossEncoder`` for the local rerank adapter.

Hand it to ``CrossEncoderRerankFactory(config, loader=...)``: no model is
downloaded or run.
"""

from __future__ import annotations

from typing import Any


class FakeCrossEncoder:
    """Scores each ``(query, passage)`` pair; counts its ``predict`` calls.

    With ``relevant`` (the synthetic corpus's answer entity, "Depot", by
    default) a passage naming it scores 1.0 and any other 0.0; with
    ``relevant=None`` a passage scores the share of the query's words it
    contains. ``kwargs`` keeps what the loader was given.
    """

    def __init__(
        self, model_id: str, *, relevant: str | None = "Depot", **kwargs: Any
    ) -> None:
        self.model_id = model_id
        self.relevant = relevant
        self.kwargs = kwargs
        self.calls = 0

    def predict(self, pairs: list[tuple[str, str]], **_: Any) -> list[float]:
        self.calls += 1
        if self.relevant is not None:
            return [float(self.relevant in passage) for _, passage in pairs]
        query = pairs[0][0].lower().split() if pairs else []
        return [
            sum(word in passage.lower() for word in query) / max(len(query), 1)
            for _, passage in pairs
        ]
