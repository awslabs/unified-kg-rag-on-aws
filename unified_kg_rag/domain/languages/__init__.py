# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which language a text is in, as far as its script tells.

Pure: letters are counted by the script table of ``shared.utils.scripts``.
"""

from unified_kg_rag.domain.languages.detection import dense_language

__all__ = ["dense_language"]
