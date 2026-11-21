# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The custom text analyzers the OpenSearch indexer defines in its indices.

A text field's mapping names its analyzer (``language_analyzers`` /
``default_analyzer``). A built-in or plugin analyzer (``english``, ``cjk``,
``nori``) needs nothing more; a name in :data:`CUSTOM_ANALYZERS` is defined
by the indexer in the ``analysis`` settings of each index whose fields use
it (:func:`with_analysis`), from built-in components only, so it works on any
OpenSearch cluster without a plugin.

- ``cjk_script_split``: OpenSearch's ``cjk`` analyzer (standard tokenizer,
  ``cjk_width``, ``lowercase``, ``cjk_bigram``) with a Hangul run split from
  what is written next to it (a ``pattern_replace`` char filter puts a space
  between a Hangul letter and a character that is neither a space nor a Han,
  Hangul or Kana letter). The standard tokenizer (UAX #29) keeps "AWS를" as
  one token, which ``cjk_bigram`` leaves whole (``aws를``), so a lexical
  query for "S3", or for "AWS" beside a Korean word, found no Korean text
  that writes a particle after the term; this analyzer yields ``aws``,
  ``를`` there, as ``nori`` and the embedded engine's ``cjk`` do. Han and
  Kana need no split (the tokenizer already breaks them from Latin letters
  and digits), and a Hangul run next to them stays one run for
  ``cjk_bigram``, as in ``cjk``. Unlike ``cjk`` it keeps English stop words;
  it folds full-width letters as ``cjk`` does (``ＡＢＣ`` is ``abc``), and the
  embedded engine's ``cjk_script_split`` does both the same. It still splits
  a combining mark written after a Hangul letter off the run, where the
  embedded engine does not (Korean text writes none). The CDK deployment maps Korean to it
  (``-c opensearch_korean_analyzer``).
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any

from unified_kg_rag.shared.utils.scripts import SCRIPTS, Spacing, script_named

__all__ = ["CJK_SCRIPT_SPLIT", "CUSTOM_ANALYZERS", "with_analysis"]

CJK_SCRIPT_SPLIT = "cjk_script_split"


def _java_class(ranges: Iterable[tuple[int, int]]) -> str:
    """``ranges`` as the body of a Java regex character class, as OpenSearch
    reads a ``pattern_replace`` pattern."""
    return "".join(
        f"\\x{{{lo:X}}}" + ("" if lo == hi else f"-\\x{{{hi:X}}}")
        for lo, hi in sorted(ranges)
    )


# Halfwidth Katakana (and its halfwidth middle dot and sound marks). The
# char filter runs before cjk_width folds it to full width, and the
# embedded engine folds width before it splits, so to match it the split
# reads these as Kana too: "ｶﾀｶﾅ한글" kept its ナ한 bigram there and lost it
# here.
_HALFWIDTH_KANA = (0xFF65, 0xFF9F)
# The Hangul letters, and every letter of the dense scripts (Han, Hangul,
# Kana), from the one script table (shared.utils.scripts) the embedded
# engine splits its dense runs by.
_HANGUL = _java_class(script_named("hangul").ranges)
_DENSE = _java_class(
    [
        *(
            span
            for script in SCRIPTS
            if script.spacing is Spacing.DENSE
            for span in script.ranges
        ),
        _HALFWIDTH_KANA,
    ]
)
# A space between a Hangul letter and a character next to it that is neither
# a space nor a dense-script letter, either way round: the embedded engine's
# split of a dense run from what is written next to it. Han and Kana need no
# split (the standard tokenizer breaks them from other letters), and Hangul
# next to them stays one run for cjk_bigram, as in cjk.
_HANGUL_BOUNDARY = (
    f"(?<=[^\\s{_DENSE}])(?=[{_HANGUL}])|(?<=[{_HANGUL}])(?=[^\\s{_DENSE}])"
)

_SPLIT_FILTER = f"{CJK_SCRIPT_SPLIT}_boundary"

# Each custom analyzer's ``analysis`` settings, by analyzer name: the
# analyzer and the char filters, tokenizers and token filters it defines.
CUSTOM_ANALYZERS: Mapping[str, Mapping[str, Mapping[str, Any]]] = MappingProxyType(
    {
        CJK_SCRIPT_SPLIT: {
            "char_filter": {
                _SPLIT_FILTER: {
                    "type": "pattern_replace",
                    "pattern": _HANGUL_BOUNDARY,
                    "replacement": " ",
                }
            },
            "analyzer": {
                CJK_SCRIPT_SPLIT: {
                    "type": "custom",
                    "char_filter": [_SPLIT_FILTER],
                    "tokenizer": "standard",
                    "filter": ["cjk_width", "lowercase", "cjk_bigram"],
                }
            },
        }
    }
)


def _field_analyzers(properties: Mapping[str, Any]) -> set[str]:
    """The analyzers the text fields of ``properties`` (and of their
    objects') name, for indexing or search."""
    names: set[str] = set()
    for field in properties.values():
        if not isinstance(field, Mapping):
            continue
        for key in ("analyzer", "search_analyzer"):
            if isinstance(name := field.get(key), str):
                names.add(name)
        names |= _field_analyzers(field.get("properties") or {})
        names |= _field_analyzers(field.get("fields") or {})
    return names


def with_analysis(
    settings: Mapping[str, Any], properties: Mapping[str, Any]
) -> dict[str, Any]:
    """``settings`` with the ``analysis`` definitions of the custom analyzers
    ``properties`` use; a definition ``settings`` already holds under the
    same name (``index_settings.analysis``) is kept."""
    used = sorted(_field_analyzers(properties) & set(CUSTOM_ANALYZERS))
    if not used:
        return dict(settings)
    analysis: dict[str, Any] = copy.deepcopy(dict(settings.get("analysis") or {}))
    for name in used:
        for kind, definitions in CUSTOM_ANALYZERS[name].items():
            section = analysis.setdefault(kind, {})
            for key, definition in definitions.items():
                section.setdefault(key, copy.deepcopy(dict(definition)))
    return {**settings, "analysis": analysis}
