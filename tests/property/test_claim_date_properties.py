# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property-based tests for claim dates (``claim_date_value``).

A coarse period reads as its first day as a start date and as its last day
as an end date; both are calendar dates a range filter orders, the end never
before the start, and a stored value reads (and keys) as itself.
"""

from __future__ import annotations

import calendar
from datetime import date

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from unified_kg_rag.domain.ingestion.claim_resolver import (
    claim_date_key,
    claim_date_value,
)

pytestmark = pytest.mark.property

_years = st.integers(min_value=1950, max_value=2100)


@st.composite
def _periods(draw: st.DrawFn) -> tuple[str, date, date]:
    """A coarse period as an extraction may write it, with its first and
    last day."""
    year = draw(_years)
    kind = draw(st.sampled_from(["year", "month", "quarter", "week"]))
    if kind == "year":
        return f"{year}", date(year, 1, 1), date(year, 12, 31)
    if kind == "month":
        month = draw(st.integers(min_value=1, max_value=12))
        last = date(year, month, calendar.monthrange(year, month)[1])
        name = draw(
            st.sampled_from(
                [
                    calendar.month_name[month],
                    calendar.month_abbr[month],
                    calendar.month_abbr[month] + ".",
                ]
            )
        )
        text = draw(
            st.sampled_from(
                [
                    f"{name} {year}",
                    f"{name}, {year}",
                    f"{name},{year}",
                    f"{year}-{month:02d}",
                    f"{year}/{month}",
                ]
            )
        )
        return text, date(year, month, 1), last
    if kind == "quarter":
        quarter = draw(st.integers(min_value=1, max_value=4))
        month = 3 * quarter - 2
        last = date(year, month + 2, calendar.monthrange(year, month + 2)[1])
        text = draw(
            st.sampled_from(
                [
                    f"Q{quarter} {year}",
                    f"q{quarter}-{year}",
                    f"Q{quarter}, {year}",
                    f"{year}-Q{quarter}",
                    f"{year} Q{quarter}",
                ]
            )
        )
        return text, date(year, month, 1), last
    week = draw(st.integers(min_value=1, max_value=52))
    monday = date.fromisocalendar(year, week, 1)
    text = draw(st.sampled_from([f"{year}-W{week:02d}", f"{year}w{week:02d}"]))
    return text, monday, date.fromisocalendar(year, week, 7)


# Examples a random run found: an abbreviation's dot before the comma.
_JAN_1950 = ("Jan., 1950", date(1950, 1, 1), date(1950, 1, 31))
_YEAR_1950 = ("1950", date(1950, 1, 1), date(1950, 12, 31))


@settings(max_examples=500, deadline=None)
@given(_periods())
@example(_JAN_1950)
@example(("Sept., 2024", date(2024, 9, 1), date(2024, 9, 30)))
@example(("Jan.,2024", date(2024, 1, 1), date(2024, 1, 31)))
@example(("Jan . 2024", date(2024, 1, 1), date(2024, 1, 31)))
def test_a_period_starts_on_its_first_day_and_ends_on_its_last(
    period: tuple[str, date, date],
) -> None:
    text, first, last = period
    assert claim_date_value(text) == first.isoformat()
    assert claim_date_value(text, end=True) == last.isoformat()
    # Stored values read, and key, as themselves in their mode.
    for value, end in ((first.isoformat(), False), (last.isoformat(), True)):
        assert claim_date_value(value, end=end) == value
        assert claim_date_key(value, end=end) == claim_date_key(text, end=end)


@given(_years, st.integers(1, 52), st.integers(1, 7))
def test_an_iso_week_with_its_weekday_is_that_day_in_either_mode(
    year: int, week: int, weekday: int
) -> None:
    day = date.fromisocalendar(year, week, weekday).isoformat()
    text = f"{year}-W{week:02d}-{weekday}"
    assert claim_date_value(text) == claim_date_value(text, end=True) == day
