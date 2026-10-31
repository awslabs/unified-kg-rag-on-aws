# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property-based tests for claim dates (``claim_date_value``) and the ids
an earlier reading of them gave a claim (``earlier_claim_ids``).

A coarse period reads as its first day as a start date and as its last day
as an end date; both are calendar dates a range filter orders, the end never
before the start, and a stored value reads (and keys) as itself. A claim
stored under an earlier reading of its dates is found under the ids
``earlier_claim_ids`` lists, whichever the reading.
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
    earlier_claim_ids,
    resolved_claim_id,
)
from unified_kg_rag.domain.models import Claim
from unified_kg_rag.shared.utils import generate_stable_id

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


@settings(max_examples=500, deadline=None)
@given(_periods(), _periods(), st.sampled_from(["as written", "end first day"]))
@example(_YEAR_1950, _JAN_1950, "end first day")
@example(_YEAR_1950, _JAN_1950, "as written")
def test_a_claim_stored_under_an_earlier_reading_is_found(
    start: tuple[str, date, date],
    end: tuple[str, date, date],
    reading: str,
) -> None:
    # The statement as extracted, and the claim it resolves to now.
    statement = Claim(
        id="raw",
        subject_id="",
        subject_name="Vendor",
        object_name="Buyer",
        type="OBLIGATION",
        start_date=start[0],
        end_date=end[0],
    )
    claim = statement.model_copy(
        update={
            "subject_id": "e-vendor",
            "object_id": "e-buyer",
            "type": "obligation",
            "start_date": claim_date_value(start[0]),
            "end_date": claim_date_value(end[0], end=True),
        }
    )
    current = resolved_claim_id(claim)

    # The id an earlier build keyed it on: its dates as written (casefolded),
    # or its end date as its first day.
    def keyed(start_key: str, end_key: str) -> str:
        return generate_stable_id(
            f"claim:e-vendor:e-buyer:obligation:{start_key}:{end_key}"
        )

    earlier = (
        keyed(" ".join(start[0].split()).casefold(), end[0].casefold())
        if reading == "as written"
        else keyed(start[1].isoformat(), end[1].isoformat())
    )
    found = earlier_claim_ids(claim, [statement])
    assert current not in found
    assert earlier in found
    # Only the statements of its dates count: one of other dates adds none.
    other = statement.model_copy(update={"start_date": "1900-01-01"})
    assert earlier_claim_ids(claim, [other]) == set()
