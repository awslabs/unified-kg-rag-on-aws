# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whether a run's failed items show more failures than
``indexing.max_failure_rate`` tolerates.

The rate is read as the tolerated chance that one item fails (a report the
model cut off, a write the store rejected). Comparing the observed share
with it made the gate depend on how many items a run has: a five-document
delta failed its community detection with 4 of 11 reports cut off (36%)
where the full build of the same corpus passed with 6 of 59 (10%), at the
same per-report chance (the ARCH baseline's local run). Counts that small
cannot tell noise from a cause.

The gate is a one-sided exact binomial test instead: it fails when, at the
tolerated chance, at least that many failures among that many items would
happen less often than :data:`FAILURE_RATE_SIGNIFICANCE`. On many items it
fails close to the rate itself (from 222 of 1,000 at 0.2); on few it needs
more evidence (4 of 11 is 16% likely at 0.2 and passes, 6 of 11 is 1% likely
and fails; 18 of 59 fails), and a run whose every item failed fails from 2
items at the default (0.2^2 = 4%). A failure the gate lets through is not
lost: the failed items are retried by the next run, as below the rate.

The community detection stage applies it to its reports, whose failures are
chance events of the model (output cut off or malformed one time in so
many). The indexing stage's write gates keep the share: a store rejects an
item on every retry or none (an outage fails nearly all), and the commit
that records the run's documents mirrors that gate
(``IncrementalIndexer._delta_writes_succeeded``), so it decides what a
partly failed commit records rather than whether noise failed the stage.
"""

from __future__ import annotations

import math

# One-sided significance of the test: the conventional 5% level. Lower it to
# tolerate more failures on small runs, raise it to tolerate fewer.
FAILURE_RATE_SIGNIFICANCE = 0.05


def failure_rate_exceeded(failed: int, total: int, max_rate: float) -> bool:
    """Whether ``failed`` of ``total`` items show a failure rate above
    ``max_rate`` (see the module docstring): the binomial chance of at
    least ``failed`` failures among ``total`` items, each failing with
    ``max_rate``, is below :data:`FAILURE_RATE_SIGNIFICANCE`. ``max_rate``
    1.0 tolerates every failure and 0.0 none."""
    if failed <= 0 or total <= 0:
        return False
    failed = min(failed, total)
    if max_rate >= 1.0:
        return False
    if max_rate <= 0.0:
        return True
    if failed <= total * max_rate:
        # At or under the expected count: the tail is over one half.
        return False
    return _upper_tail(failed, total, max_rate) < FAILURE_RATE_SIGNIFICANCE


def _upper_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), summed in log space (no large
    integers), largest terms first so it stops once they are negligible."""
    log_p, log_q = math.log(p), math.log1p(-p)
    log_n = math.lgamma(n + 1)
    total = 0.0
    for i in range(k, n + 1):
        term = math.exp(
            log_n
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * log_p
            + (n - i) * log_q
        )
        total += term
        # Past the mode the terms only shrink: stop once one adds nothing.
        if term < total * 1e-12:
            break
    return min(total, 1.0)
