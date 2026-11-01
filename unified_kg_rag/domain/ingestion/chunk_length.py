# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""How many characters a chunk size spans (``processing.chunking.length_unit``).

With ``tokens`` the sizes count the local token estimate
(``adapters.llm.token_estimate``), which counts one token per
:data:`LATIN_CHARS_PER_TOKEN` characters outside the dense scripts and one
per dense character. The chunkers locate overlaps in characters and the
output floors of the long-output prompts size an answer by a chunk's
characters, so both take the bound from here.
"""

from __future__ import annotations

from unified_kg_rag.domain.models.config import ChunkLengthUnit

# Characters per token of the text outside the dense scripts, in the local
# token estimate.
LATIN_CHARS_PER_TOKEN = 4


def max_chars(size: int, unit: ChunkLengthUnit) -> int:
    """The most characters a text of ``size`` units can hold.

    The token estimate counts at least ``other // 4`` tokens for ``other``
    non-dense characters and one per dense character, so ``n`` tokens span
    at most ``4 * n + 3`` characters.
    """
    if unit == ChunkLengthUnit.TOKENS:
        return size * LATIN_CHARS_PER_TOKEN + LATIN_CHARS_PER_TOKEN - 1
    return size
