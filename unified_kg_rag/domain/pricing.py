# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model prices and the USD cost of model usage: the one owner.

``models.prices`` (``ModelsConfig.prices``) maps a model id to its
``ModelPrice`` (a config section, re-exported here). Every USD figure the
package computes reads it here: a ``run-eval`` comparison's cost estimate,
the console's per-answer cost and dashboard, an ingestion run's ``processing.max_run_usd`` and
``run-ingestion --estimate``. No prices ship with the package: one would go
stale. An embedding model is priced by its input tokens when its entry sets
``embedding_per_million_tokens``; without one its requests are left out
(``unpriced_embedding_models`` names it) rather than making the cost
unknown.

A price is found by the model id the usage records name (the id the model
factory resolved, an inference profile included), else by the entries of
the same model under another inference-profile prefix (``global.``,
``us.``, ...) or none, when they all agree: the configured model ids an
estimate reads are bare, the ids a run records resolved.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from unified_kg_rag.domain.models.config import ModelPrice, bare_model_id

if TYPE_CHECKING:
    from unified_kg_rag.domain.models.usage import LLMUsage, LLMUsageCounts

__all__ = [
    "ModelPrice",
    "counts_cost_usd",
    "price_of",
    "snapshot_cost_usd",
    "token_rates",
    "tokens_cost_usd",
    "unpriced_embedding_models",
    "unpriced_models",
    "usage_cost_usd",
]


def price_of(prices: Mapping[str, ModelPrice], model_id: str) -> ModelPrice | None:
    """The price of ``model_id``: its own entry, else the one price every
    entry of the same model under another inference-profile prefix agrees
    on; ``None`` without one."""
    if (price := prices.get(model_id)) is not None:
        return price
    bare = bare_model_id(model_id)
    same = [p for key, p in prices.items() if bare_model_id(key) == bare]
    return same[0] if same and all(p == same[0] for p in same) else None


def token_rates(price: ModelPrice | None) -> tuple[float, float] | None:
    """A price's (input, output) USD per million tokens; ``None`` when it
    has none (a rerank model's, or no price)."""
    if price is None:
        return None
    rates = (price.input_per_million_tokens, price.output_per_million_tokens)
    if rates[0] is None or rates[1] is None:
        return None
    return rates[0], rates[1]


def tokens_cost_usd(price: ModelPrice, input_tokens: int, output_tokens: int) -> float:
    """The USD of ``input_tokens`` and ``output_tokens`` at ``price``'s
    token rates (no cache reads or writes: an estimate's upper bound)."""
    rate, output = token_rates(price) or (0.0, 0.0)
    return (max(0, input_tokens) * rate + max(0, output_tokens) * output) / 1e6


def counts_cost_usd(counts: LLMUsageCounts, price: ModelPrice) -> float:
    """The USD of one model's counts: ``input_tokens`` includes the cache
    reads and writes, each priced at its own rate when ``price`` has one;
    a rerank model's search units and an embedding model's input tokens
    at their own prices."""
    cached = counts.cache_read_input_tokens + counts.cache_write_input_tokens
    rate, output = token_rates(price) or (0.0, 0.0)
    read = price.cache_read_per_million_tokens
    write = price.cache_write_per_million_tokens
    units = price.per_thousand_rerank_search_units or 0.0
    embedding = price.embedding_per_million_tokens or 0.0
    return (
        max(0, counts.input_tokens - cached) * rate
        + counts.cache_read_input_tokens * (rate if read is None else read)
        + counts.cache_write_input_tokens * (rate if write is None else write)
        + counts.output_tokens * output
        + counts.embedding_input_tokens * embedding
    ) / 1e6 + counts.rerank_search_units * units / 1e3


def unpriced_models(usage: LLMUsage, prices: Mapping[str, ModelPrice]) -> set[str]:
    """The models ``usage`` called that ``prices`` cannot price: a language
    model without token rates, a reranker without a search-unit price."""
    unpriced: set[str] = set()
    for model, counts in usage.by_model.items():
        price = price_of(prices, model)
        if counts.llm_calls and token_rates(price) is None:
            unpriced.add(model)
        elif counts.rerank_calls and (
            price is None or price.per_thousand_rerank_search_units is None
        ):
            unpriced.add(model)
    return unpriced


def unpriced_embedding_models(
    usage: LLMUsage, prices: Mapping[str, ModelPrice]
) -> set[str]:
    """The embedding models ``usage`` sent requests to that have no
    ``embedding_per_million_tokens``: their requests are left out of
    :func:`usage_cost_usd` (which they do not make ``None``, so a price
    list written before embeddings were counted still prices the rest)."""
    return {
        model
        for model, counts in usage.by_model.items()
        if counts.embedding_calls
        and (
            (price := price_of(prices, model)) is None
            or price.embedding_per_million_tokens is None
        )
    }


def usage_cost_usd(
    usage: LLMUsage, prices: Mapping[str, ModelPrice]
) -> tuple[float | None, set[str]]:
    """The USD cost of ``usage`` (an answer's, a run's), and the models
    without a price (the cost is then ``None``)."""
    if unpriced := unpriced_models(usage, prices):
        return None, unpriced
    usd = 0.0
    for model, counts in usage.by_model.items():
        if (price := price_of(prices, model)) is not None:
            usd += counts_cost_usd(counts, price)
    return float(usd), set()


def snapshot_cost_usd(
    usage: Mapping[str, Any] | None, prices: Mapping[str, ModelPrice]
) -> tuple[float | None, list[str]]:
    """:func:`usage_cost_usd` of a usage snapshot (``metadata["usage"]``, a
    ``UsageMeter.snapshot()``): its USD (``None`` with a model unpriced, or
    no readable snapshot) and the models it could not price, sorted: those
    that make it ``None`` and the embedding models whose requests it leaves
    out (:func:`unpriced_embedding_models`)."""
    from unified_kg_rag.domain.models.usage import LLMUsage

    if not usage:
        return None, []
    try:
        spent = LLMUsage.model_validate(dict(usage))
    except ValidationError:
        return None, []
    usd, unpriced = usage_cost_usd(spent, prices)
    return usd, sorted(unpriced | unpriced_embedding_models(spent, prices))
