# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``search.hybrid.lexical_fuzziness`` on a real OpenSearch: what it trades.

Synthetic name lookups (pronounceable made-up organization names, a fifth of
them with a second organization one edit away, as "Acme" and "Acne"):
queries name an organization exactly or with one typo. A dev split and a
held-out split (separate corpora and queries) give the numbers in
docs/concepts/defaults-and-evidence.md#lexical-fuzziness; the test pins
their shape on both: fuzziness finds the typos and also returns near-miss
names, exact matching the reverse. The embedded engine matches exactly
whatever the setting says, so this runs on the OpenSearch container only.
Skipped unless ``LOCAL_STORES=1``.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from tests.fixtures.store_backends import open_harness
from unified_kg_rag.domain.models import (
    RetrieverRole,
    SearchQuery,
    SearchType,
    TextUnit,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.local_stores,
]

_ONSETS = "b c d f g h k l m n p r s t v z br dr gr kr pl st tr".split()
_VOWELS = "aeiou"
_FILLER = (
    "supplies parts to the buyer under a contract signed in spring with "
    "delivery terms penalties and a warranty for each shipment"
).split()
_TOP = 10


def _name(rng: random.Random) -> str:
    return "".join(
        rng.choice(_ONSETS) + rng.choice(_VOWELS) for _ in range(rng.randint(2, 4))
    )


def _edit(rng: random.Random, word: str) -> str:
    """``word`` with one letter substituted, deleted or inserted."""
    i = rng.randrange(len(word))
    letter = rng.choice("abcdefghiklmnoprstuvz")
    match rng.choice("sdi"):
        case "s":
            return word[:i] + letter + word[i + 1 :]
        case "d" if len(word) > 3:
            return word[:i] + word[i + 1 :]
        case _:
            return word[:i] + letter + word[i:]


def _split(seed: int) -> tuple[list[TextUnit], list[tuple[str, str, str]]]:
    """150 organizations and 30 near-miss neighbours, one document each, and
    60 exact and 40 one-typo queries as (kind, query, document id)."""
    rng = random.Random(seed)
    names: list[str] = []
    while len(names) < 150:
        if (name := _name(rng)) not in names:
            names.append(name)
    for base in rng.sample(names, 30):
        for _ in range(20):
            other = _edit(rng, base)
            if other not in names:
                names.append(other)
                break
    ids = {name: f"s{seed}-{i:03d}" for i, name in enumerate(names)}
    units = [
        TextUnit(
            id=ids[name], text=f"{name.capitalize()} {' '.join(rng.sample(_FILLER, 8))}"
        )
        for name in names
    ]
    queries = [
        ("exact", name.capitalize(), ids[name]) for name in rng.sample(names, 60)
    ]
    for name in rng.sample(names, 40):
        for _ in range(20):
            typo = _edit(rng, name)
            if typo not in ids:
                queries.append(("typo", typo.capitalize(), ids[name]))
                break
    return units, queries


def _measure(
    tmp_path: Path, seed: int, settings: tuple[str, ...]
) -> dict[str, dict[str, dict[str, float]]]:
    """hit@1 and the wrong hits in the top 10 per query kind, per setting."""
    units, queries = _split(seed)
    opened = open_harness("container", tmp_path)
    try:
        harness = next(opened)
        harness.initialize()
        stats = harness.vector_indexer.index_text_units(units)
        assert stats.failed_items == 0, stats.errors
        prefix = harness.config.indexing.opensearch.text_units_index_prefix
        report: dict[str, dict[str, dict[str, float]]] = {}
        for fuzziness in settings:
            searcher = harness.derive(
                lambda config, value=fuzziness: setattr(
                    config.search.hybrid, "lexical_fuzziness", value
                )
            )
            results = searcher.retrieve(
                RetrieverRole.DOCUMENT,
                *(
                    SearchQuery(
                        query=query,
                        search_type=SearchType.LEXICAL,
                        index_prefixes=prefix,
                        top_k=_TOP,
                        retrieval_multiplier=1,
                    )
                    for _, query, _ in queries
                ),
            )
            rows: dict[str, list[tuple[bool, int]]] = {"exact": [], "typo": []}
            for (kind, _, wanted), hits in zip(queries, results, strict=True):
                found = [str(hit.source) for hit in hits][:_TOP]
                rows[kind].append(
                    (found[:1] == [wanted], sum(f != wanted for f in found))
                )
            report[fuzziness] = {
                kind: {
                    "hit@1": sum(first for first, _ in kind_rows) / len(kind_rows),
                    "wrong@10": sum(wrong for _, wrong in kind_rows) / len(kind_rows),
                }
                for kind, kind_rows in rows.items()
            }
        return report
    finally:
        opened.close()


@pytest.mark.parametrize("seed", [11, 23], ids=["dev", "held-out"])
def test_fuzziness_trades_near_miss_names_for_typos(tmp_path: Path, seed: int) -> None:
    report = _measure(tmp_path, seed, ("AUTO", "0"))
    fuzzy, exact = report["AUTO"], report["0"]
    # Exact matching finds every exactly named organization first, with no
    # other name, and no typo.
    assert exact["exact"] == {"hit@1": 1.0, "wrong@10": 0.0}
    assert exact["typo"]["hit@1"] == 0.0
    # AUTO finds nearly every typo, and returns near-miss names beside the
    # exact ones (now and then ranked first).
    assert fuzzy["typo"]["hit@1"] >= 0.9
    assert fuzzy["exact"]["wrong@10"] > 0.2
    assert fuzzy["exact"]["hit@1"] < 1.0
