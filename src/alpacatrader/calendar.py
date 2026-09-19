"""The NYSE trading calendar, derived from the store rather than declared.

§3 lists the calendar as a new module: holidays, half days, and the DST change that moves the UTC
time of the session twice a year. This takes all three from the bars themselves.

**Measured, not tabulated.** A hard-coded holiday table is a second source of truth that goes stale
the year the exchange adds a closure, and it can disagree with the data it is supposed to describe —
silently, because nothing compares them. A session is a day on which the basket traded, a holiday is
a weekday on which it did not, and a half day is a session that stopped early. All three fall out of
the index, which is the same index every downstream window is counted on.

The cost of deriving it is that a day the store is simply *missing* looks like a holiday. That is why
`sessions` is built from the union across symbols and why `suspicious` exists: a real closure closes
the whole basket, so a day where a handful of names traded and the rest did not is a gap in the
store, not a holiday, and it has to be visible as a gap.
"""

from __future__ import annotations

from datetime import time

import pandas as pd

from alpacatrader.data.candles import CLOSE, EXCHANGE, OPEN

FULL_SESSION = 390  # minutes from 09:30 to 16:00
HALF_SESSION = 210  # the 13:00 close of the day after Thanksgiving, Christmas Eve, July 3


def sessions(index: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per trading day: when it opened, when its last bar opened, how many bars it held."""
    local = pd.DatetimeIndex(index).tz_convert(EXCHANGE)
    by_day = pd.Series(local, index=local.date).groupby(level=0)
    out = pd.DataFrame(
        {
            "open": by_day.min().dt.time,
            "close": by_day.max().dt.time,
            "bars": by_day.size(),
            "utc_open": pd.Series(pd.DatetimeIndex(index), index=local.date).groupby(level=0).min().dt.time,
        }
    )
    out.index = pd.to_datetime(out.index)
    return out.assign(half_day=out.close < time(15, 0))


def holidays(days: pd.DataFrame) -> pd.DatetimeIndex:
    """Weekdays inside the covered span on which the basket did not trade at all."""
    span = pd.date_range(days.index.min(), days.index.max(), freq="B")
    return span.difference(days.index)


def suspicious(days: pd.DataFrame, expected: int = FULL_SESSION) -> pd.DataFrame:
    """Sessions that are neither a full day nor a clean half day — a gap in the store, most likely.

    A real half day stops at 13:00 and holds 210 minutes. A session holding 380 of 390 is not a
    short session, it is ten minutes the store does not have, and calling it a calendar fact would
    bury it. `bars` counts the *union* across symbols, so this is about the basket and not one name.
    """
    full = (~days.half_day) & (days.bars < expected)
    short = days.half_day & (days.bars != HALF_SESSION)
    return days[full | short]


def dst_changes(days: pd.DataFrame) -> pd.DataFrame:
    """The sessions where the UTC opening time moved — twice a year, and the session never did."""
    moved = days.utc_open != days.utc_open.shift(1)
    return days[moved & days.utc_open.shift(1).notna()]


def _selfcheck() -> None:
    """The three calendar facts, on an index built here — the store needs the network."""

    def day(date: str, minutes: int) -> pd.DatetimeIndex:
        return pd.date_range(f"{date} 09:30", periods=minutes, freq="1min", tz=EXCHANGE).tz_convert("UTC")

    # One week in January: Monday full, Tuesday closed, Wednesday full, Thursday a half day,
    # Friday ten minutes short of a full one. A week and not six months, so `holidays` is asked
    # about a span whose answer can be written out.
    index = day("2026-01-05", FULL_SESSION)
    index = index.append(day("2026-01-07", FULL_SESSION))
    index = index.append(day("2026-01-08", HALF_SESSION))
    index = index.append(day("2026-01-09", 380))

    days = sessions(index)
    assert len(days) == 4
    assert (days.open == OPEN).all(), "every session opens at the bell, whatever UTC says"
    assert days.close.iloc[0] == time(15, 59)

    # Trap 2: a half day is a session that stops at 13:00, and it needs no rule of its own.
    assert list(days.half_day) == [False, False, True, False]
    assert days.close.iloc[2] == time(12, 59) and days.bars.iloc[2] == HALF_SESSION

    # A weekday with no bars is a closure; a weekend never appears and is not one.
    assert list(holidays(days).strftime("%Y-%m-%d")) == ["2026-01-06"], list(holidays(days))
    assert not any(d.dayofweek >= 5 for d in holidays(days)), "a weekend is not a holiday"

    # And a session ten minutes short is a gap in the store, not a calendar fact.
    odd = suspicious(days)
    assert list(odd.index.strftime("%Y-%m-%d")) == ["2026-01-09"], f"got {list(odd.index)}"

    # Trap 1, on its own span: the UTC opening time moves with DST and the exchange time does not.
    across = sessions(day("2026-03-06", FULL_SESSION).append(day("2026-03-12", FULL_SESSION)))
    assert (across.open == OPEN).all(), "the session does not move"
    assert across.utc_open.iloc[0] == time(14, 30), "before the change, UTC-5"
    assert across.utc_open.iloc[1] == time(13, 30), "after it, UTC-4"
    assert len(dst_changes(across)) == 1 and dst_changes(across).index[0].strftime("%m-%d") == "03-12"
    assert len(dst_changes(days)) == 0, "a week inside one offset moves nothing"

    assert CLOSE == time(16, 0) and FULL_SESSION == 390


if __name__ == "__main__":
    _selfcheck()
    print("ok — holidays, half days and the DST move all read off the bars")
