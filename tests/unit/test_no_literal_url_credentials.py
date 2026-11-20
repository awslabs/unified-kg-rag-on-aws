# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""No tracked file holds a URL with a literal password (``s://u:<password>@h``).

Secret scanners report one as a leaked password (GitLab's "Password in URL"
rule flagged the redaction tests' made-up ones, from the commits that added
them, after their lines were marked ``gitleaks:allow``). A test builds its
URL with ``tests.fixtures.url_credentials.userinfo_url`` instead; a document
writes the password as a placeholder (``<password>``). The
``no-literal-url-credentials`` pre-commit hook greps a commit's files for the
same pattern.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404 - runs git on a fixed argv
from pathlib import Path

import pytest
import yaml

from tests.fixtures.url_credentials import (
    FAKE_PASSWORD,
    FAKE_USER,
    LITERAL_USERINFO,
    userinfo_url,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="needs a git checkout",
)
def test_no_tracked_file_holds_a_literal_url_credential() -> None:
    tracked = subprocess.run(  # nosec B603 B607 - fixed argv
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    assert len(tracked) > 100
    found = []
    for name in filter(None, tracked):
        path = REPO_ROOT / name
        try:
            text = path.read_text(encoding="utf-8")
        except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError):
            continue  # deleted in the work tree, a submodule, or binary
        for number, line in enumerate(text.splitlines(), start=1):
            if LITERAL_USERINFO.search(line):
                found.append(f"{name}:{number}")
    assert found == [], (
        "literal URL credentials (build them with "
        f"tests.fixtures.url_credentials.userinfo_url): {found}"
    )


def test_the_pre_commit_hook_greps_for_the_same_pattern() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    (hook,) = [
        hook
        for repo in config["repos"]
        for hook in repo["hooks"]
        if hook["id"] == "no-literal-url-credentials"
    ]
    assert hook["language"] == "pygrep"
    assert hook["entry"] == LITERAL_USERINFO.pattern


def _planted(user: str, password: str, rest: str, scheme: str = "https") -> str:
    # Joined here so this file holds no literal either.
    return scheme + "://" + user + ":" + password + "@" + rest


@pytest.mark.parametrize(
    "line",
    [
        # The forms GitLab's rule reported on the redaction tests.
        _planted("alice", "S3cretTok", "127.0.0.1:9/v1?api-key=QQQ", "http"),
        _planted("user", "secret", "seed-bucket/procure.jsonl", "s3"),
        _planted("reader", "SECRET", "schemas.example/trade"),
        # Shorter than GitLab's rule matches, as detect-secrets reports them.
        _planted("u", "p", "[2001:db8::7]/v1", "http"),
        _planted("", "pw", "redis.example:6379", "redis"),
        _planted("app_user", "kJ8mN2pQ9rT5", "db.example.internal", "postgres"),
        f'url = "{_planted("user", "token", "example.org/o.ttl")}"',
    ],
)
def test_the_guard_matches_a_literal_credential(line: str) -> None:
    assert LITERAL_USERINFO.search(line)


@pytest.mark.parametrize(
    "line",
    [
        "https://example.org/x?user=a:b@c",
        "https://reader@example.org/seed.jsonl",
        "ssh://git@example.org:22/repo.git",
        "git@example.org:org/repo.git",
        _planted("user", "<password>", "db.example"),
        _planted("user", "${DB_PASSWORD}", "db.example"),
        _planted("user", "{password}", "db.example"),
        _planted("user", "%DB_PASSWORD%", "db.example"),
        'return f"{scheme}{separator}{user}:{password}@{rest}"',
    ],
)
def test_the_guard_leaves_urls_without_a_literal_password(line: str) -> None:
    assert not LITERAL_USERINFO.search(line)


def test_userinfo_url_puts_the_credential_after_the_scheme() -> None:
    assert userinfo_url("https://example.org:8443/x?k=v#f") == _planted(
        FAKE_USER, FAKE_PASSWORD, "example.org:8443/x?k=v#f"
    )
    assert userinfo_url("http://[::1]/v1", user="u", password="p") == _planted(
        "u", "p", "[::1]/v1", "http"
    )
    assert LITERAL_USERINFO.search(userinfo_url("s3://bucket/key"))
    for bad in ("example.org/x", userinfo_url("https://example.org/x")):
        with pytest.raises(ValueError, match="not a URL without userinfo"):
            userinfo_url(bad)
