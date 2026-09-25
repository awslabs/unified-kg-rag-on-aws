# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the optional parsers and their patched URL handling without AWS."""

from __future__ import annotations

import socket

import pytest

pytest.importorskip("unstructured")

from langchain_community.document_loaders import (  # noqa: E402
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from unstructured.partition.html import partition_html  # noqa: E402
from unstructured.partition.md import partition_md  # noqa: E402
from unstructured.safe_http import UnsafeURLError  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("partition", [partition_html, partition_md])
@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1/", "http://169.254.169.254/", "http://[::1]/"],
)
def test_private_url_rejected_before_connection(partition, url, monkeypatch):
    monkeypatch.delenv("UNSTRUCTURED_ALLOW_PRIVATE_URL", raising=False)

    def unexpected_connection(*args, **kwargs):
        pytest.fail("Private URL validation must happen before any connection")

    monkeypatch.setattr(socket.socket, "connect", unexpected_connection)
    with pytest.raises(UnsafeURLError):
        partition(url=url)


@pytest.mark.parametrize(
    ("loader", "suffix", "content"),
    [
        (UnstructuredMarkdownLoader, ".md", "# Security smoke test\n"),
        (UnstructuredHTMLLoader, ".html", "<h1>Security smoke test</h1>"),
    ],
)
def test_local_document_still_parses(loader, suffix, content, tmp_path):
    source = tmp_path / f"document{suffix}"
    source.write_text(content, encoding="utf-8")
    documents = loader(str(source)).load()
    assert "Security smoke test" in "\n".join(d.page_content for d in documents)
