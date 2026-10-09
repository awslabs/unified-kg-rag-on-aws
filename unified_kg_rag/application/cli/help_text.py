# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared ``--help`` epilog text for the ``run-*`` CLIs."""

from __future__ import annotations

USER_GUIDE_URL = (
    "https://github.com/awslabs/unified-kg-rag-on-aws/blob/main/docs/user-guide.md"
)


def doc_epilog(section: str, example: str) -> str:
    """Epilog naming the user-guide section and one example invocation.

    Use with an ``argparse`` raw formatter so the line breaks are kept.
    """
    return f"example:\n  {example}\n\ndocumentation:\n  {USER_GUIDE_URL}#{section}\n"
