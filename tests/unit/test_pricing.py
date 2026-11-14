# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One owner of model prices (``domain.pricing``, ``models.prices``).

The prices lived under ``evaluation``, so the console, the dashboard and
anything outside run-eval reached into the evaluation section for them; an
ingestion budget or estimate had no place to read them from. They are one
``models.prices`` section now, and ``evaluation.prices`` is its deprecated
alias.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from unified_kg_rag.domain.models import Config, LLMUsage
from unified_kg_rag.domain.models import config as config_package
from unified_kg_rag.domain.models.config import _root as root_module
from unified_kg_rag.domain.pricing import (
    ModelPrice,
    price_of,
    tokens_cost_usd,
    unpriced_models,
    usage_cost_usd,
)

pytestmark = pytest.mark.unit


@contextlib.contextmanager
def monkeypatched_deprecated(keys: dict[str, str]) -> Iterator[None]:
    saved = dict(config_package.DEPRECATED_KEYS)
    config_package.DEPRECATED_KEYS.clear()
    config_package.DEPRECATED_KEYS.update(keys)
    try:
        yield
    finally:
        config_package.DEPRECATED_KEYS.clear()
        config_package.DEPRECATED_KEYS.update(saved)


CONFIG_LOGGER = "unified_kg_rag.domain.models.config"
SONNET = ModelPrice(input_per_million_tokens=2.0, output_per_million_tokens=10.0)
HAIKU = ModelPrice(input_per_million_tokens=0.1, output_per_million_tokens=0.5)


def test_a_price_is_found_by_its_own_id_first() -> None:
    prices = {"global.m": SONNET, "m": HAIKU}
    assert price_of(prices, "global.m") is SONNET
    assert price_of(prices, "m") is HAIKU


def test_a_bare_id_and_a_profile_id_find_each_other_when_unambiguous() -> None:
    # An estimate reads the configured (bare) ids; a run records the
    # resolved inference profile.
    assert price_of({"global.m": SONNET}, "m") is SONNET
    assert price_of({"m": SONNET}, "us.m") is SONNET
    assert price_of({"global.m": SONNET, "us.m": SONNET}, "eu.m") == SONNET


def test_profiles_priced_apart_price_no_other_profile() -> None:
    # An in-Region profile can list another price: guessing one is wrong.
    prices = {"global.m": SONNET, "us.m": HAIKU}
    assert price_of(prices, "m") is None
    assert price_of(prices, "eu.m") is None
    assert price_of(prices, "other") is None


def test_usage_cost_names_the_unpriced_models() -> None:
    usage = LLMUsage.model_validate(
        {
            "by_model": {
                "global.m": {
                    "llm_calls": 1,
                    "input_tokens": 1000,
                    "output_tokens": 100,
                },
                "reranker": {"rerank_calls": 1, "rerank_search_units": 2},
            }
        }
    )
    assert usage_cost_usd(usage, {"m": SONNET}) == (None, {"reranker"})
    assert unpriced_models(usage, {}) == {"global.m", "reranker"}
    rerank = ModelPrice(per_thousand_rerank_search_units=2.0)
    usd, unpriced = usage_cost_usd(usage, {"m": SONNET, "reranker": rerank})
    assert unpriced == set()
    assert usd == pytest.approx((1000 * 2.0 + 100 * 10.0) / 1e6 + 2 * 2.0 / 1e3)


def test_tokens_cost_prices_input_and_output() -> None:
    assert tokens_cost_usd(SONNET, 1_000_000, 100_000) == pytest.approx(3.0)
    rerank = ModelPrice(per_thousand_rerank_search_units=2.0)
    assert tokens_cost_usd(rerank, 1000, 1000) == 0.0


def test_models_prices_is_the_section_every_figure_reads() -> None:
    config = Config(models={"prices": {"m": SONNET.model_dump()}})
    assert config.models.prices == {"m": SONNET}
    assert not hasattr(config.evaluation, "prices")


def test_evaluation_prices_loads_into_models_prices_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    root_module._warn_deprecated.cache_clear()
    with caplog.at_level(logging.WARNING, logger=CONFIG_LOGGER):
        config = Config(
            evaluation={"prices": {"m": SONNET.model_dump(), "n": HAIKU.model_dump()}},
            models={"prices": {"n": SONNET.model_dump()}},
        )
    # models.prices keeps its own entry for a model both name.
    assert config.models.prices == {"m": SONNET, "n": SONNET}
    (warned,) = [r.getMessage() for r in caplog.records]
    assert "evaluation.prices is deprecated" in warned
    assert "models.prices" in warned
    assert config_package.DEPRECATED_KEYS["evaluation.prices"] == "models.prices"
    # The config rebuilt from its input reads the same prices, and its
    # input names only the replacement: a later change to models.prices is
    # not merged with the deprecated entries again.
    assert "prices" not in config.as_input().get("evaluation", {})
    assert Config.model_validate(config.as_input()).models.prices == (
        config.models.prices
    )
    changed = config.updated({"models.prices": {"m": HAIKU.model_dump()}})
    assert changed.models.prices == {"m": HAIKU}


def test_a_deprecated_key_and_its_replacement_that_disagree_are_refused() -> None:
    # Not a mapping: nothing to merge entry by entry.
    data = {"a": {"old": 1}, "b": {"new": 2}}
    with monkeypatched_deprecated({"a.old": "b.new"}):
        with pytest.raises(ValueError, match="a.old and b.new are both set"):
            config_package.migrate_deprecated_keys(data)
        moved, keys = config_package.migrate_deprecated_keys(
            {"a": {"old": 2}, "b": {"new": 2}}
        )
        assert keys == ["a.old"] and moved == {"a": {}, "b": {"new": 2}}
    assert data == {"a": {"old": 1}, "b": {"new": 2}}


def test_an_unknown_key_beside_a_deprecated_one_is_still_refused() -> None:
    with pytest.raises(ValueError, match="evaluation.pricez"):
        Config(evaluation={"prices": {}, "pricez": {}})


def test_the_loader_loads_a_deprecated_key_from_a_file(tmp_path: Path) -> None:
    from unified_kg_rag.application.config import ConfigLoader

    path = tmp_path / "config.yaml"
    path.write_text(
        "evaluation:\n  prices:\n    m:\n      input_per_million_tokens: 2.0\n"
        "      output_per_million_tokens: 10.0\n",
        encoding="utf-8",
    )
    assert ConfigLoader(config_path=path).load_config().models.prices == {"m": SONNET}


def test_the_loader_loads_a_deprecated_variable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A variable under a deprecated key sets its replacement (a scalar here:
    # a price entry is a mapping no variable sets).
    from unified_kg_rag.application.config import ConfigLoader

    monkeypatch.setenv("UKR__SEARCH__LEGACY_TIMEOUT", "42")
    with monkeypatched_deprecated(
        {"search.legacy_timeout": "search.query_timeout_seconds"}
    ):
        with caplog.at_level(logging.WARNING):
            config = ConfigLoader().load_config()
    assert config.search.query_timeout_seconds == 42
    assert any(
        "set UKR__SEARCH__QUERY_TIMEOUT_SECONDS" in r.getMessage()
        for r in caplog.records
    )
