# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared ``--help`` epilog text for the ``run-*`` CLIs."""

from __future__ import annotations

DOCS_URL = "https://github.com/awslabs/unified-kg-rag-on-aws/blob/main/docs/"


def doc_epilog(page: str, example: str) -> str:
    """Epilog naming the documentation page and one example invocation.

    ``page`` is a path under ``docs/``, optionally with a ``#`` heading id
    (``guides/ingest.md``, ``user-guide.md#6-evaluation-run-eval``). Use with
    an ``argparse`` raw formatter so the line breaks are kept.
    """
    return f"example:\n  {example}\n\ndocumentation:\n  {DOCS_URL}{page}\n"
