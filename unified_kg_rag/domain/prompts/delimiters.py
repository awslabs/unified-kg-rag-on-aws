# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep untrusted prompt inputs inside the tags that delimit them.

Prompts wrap corpus text, retrieved context and reports in a tag pair
(``<input_text>{input_text}</input_text>``) and tell the model that everything
inside is data. A value containing ``</input_text>`` would end that block
early, and whatever followed it would read as instructions. The chain builder
neutralises the delimiter tags of a prompt in every input value before the
template is formatted. Pure functions, no backend imports.
"""

import re
from collections.abc import Iterable
from functools import lru_cache

# A tag pair whose content holds a template placeholder and no other tag:
# ``<context>\n{context}\n</context>``.
_DELIMITED_PLACEHOLDER = re.compile(
    r"<([A-Za-z_][\w\-]*)>[^<]*?(?<!\{)\{[A-Za-z_]\w*\}(?!\})[^<]*?</\1>"
)


def delimiter_tags(*templates: str) -> frozenset[str]:
    """Names of the tags that delimit a placeholder in ``templates``."""
    return frozenset(
        match.group(1)
        for template in templates
        for match in _DELIMITED_PLACEHOLDER.finditer(template)
    )


def neutralise_delimiters(value: str, tags: Iterable[str]) -> str:
    """Escape the opening and closing forms of ``tags`` inside ``value``.

    ``</context>``, ``< / Context >`` and ``<context>`` become ``&lt;/context>``
    and so on (case-insensitive, whitespace-tolerant), so the value can no
    longer end its block or open a new one; other text, including other tags,
    is kept as is.
    """
    names = frozenset(tags)
    if not names or "<" not in value:
        return value
    return _delimiter_pattern(names).sub(r"&lt;\1\2", value)


@lru_cache(maxsize=64)
def _delimiter_pattern(names: frozenset[str]) -> re.Pattern[str]:
    alternatives = "|".join(map(re.escape, sorted(names, key=len, reverse=True)))
    return re.compile(r"<(\s*/?\s*)(" + alternatives + r")(?![\w\-])", re.IGNORECASE)
