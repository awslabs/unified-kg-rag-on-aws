# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Made-up URL credentials for the redaction tests, built at run time.

A test that checks a URL's ``user:password@`` never reaches a message, a
manifest or a log needs such a URL. Written as a literal, it is what secret
scanners report as a leaked password (GitLab's "Password in URL" rule,
detect-secrets' basic-auth detector), and an inline ``gitleaks:allow``
does not help once a commit without it is in the history the scanner reads.
So a test passes the URL without credentials to :func:`userinfo_url`, and
no literal ever lands in a commit. ``LITERAL_USERINFO`` is the form the
scanners match; ``tests/unit/test_no_literal_url_credentials.py`` fails on
it in any tracked file.
"""

from __future__ import annotations

import re

FAKE_USER = "user"
FAKE_PASSWORD = "secret"

# ``scheme://user:<password>@`` with a literal password: one that does not
# start as a placeholder (``{x}``, ``${X}``, ``<password>``, ``%VAR``, ``(``)
# and holds no whitespace, ``/``, ``@``, quote, backtick or bracket. Of
# those, a URL's userinfo can hold ``'`` unescaped (an RFC 3986 sub-delim);
# the others it cannot. They are left out so a match cannot run across a
# quote or a path in source text (``'https://u:' + pw + '@h'`` is code, not
# a password), which makes a password holding ``'`` a miss the guard accepts
# (it is a tripwire). The password may hold ``:``, ``#`` and ``?``, as in
# GitLab's "Password in URL" rule. Every userinfo that rule reports matches,
# unless it holds one of the left-out characters (GitLab also reports some
# with ``/``, ``'`` or a bracket in the password, or ``?``, ``#``, ``'`` or
# ``[`` in the user part), and so do the ones it skips: passwords under 5
# characters, any host, a one-letter scheme.
LITERAL_USERINFO = re.compile(
    r"[A-Za-z][A-Za-z0-9+.-]*://[^\s:/?#\[\]@'\"`]*:(?![{$<%(])[^\s/@'\"`\[\]]+@"
)


def userinfo_url(
    url: str, *, user: str = FAKE_USER, password: str = FAKE_PASSWORD
) -> str:
    """``url`` with ``user:password@`` after its scheme: ``userinfo_url(
    "https://example.org/x", user="reader")`` is the URL with userinfo
    ``reader`` and :data:`FAKE_PASSWORD`.

    Raises:
        ValueError: ``url`` has no ``scheme://`` or already holds userinfo.
    """
    scheme, separator, rest = url.partition("://")
    if not separator or "@" in rest.partition("/")[0]:
        raise ValueError(f"not a URL without userinfo: {url!r}")
    return f"{scheme}{separator}{user}:{password}@{rest}"
