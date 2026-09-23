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
EARLY_CLOSE = time(13, 0)
AFTERNOON = 180  # minutes from the early close to the normal one

# A half day is detected by how busy its afternoon is, not by when its last bar printed. Measured
# 2026-09-19 on the built store, and this is trap 2 of §3 coming out differently from the guess in
# it: a half day does **not** simply stop at 13:00. Alpaca's SIP tape carries late and out-of-
# sequence prints for hours afterwards, so 2016-11-25 closes at 13:00 and still shows a last bar at
# 15:35, and 2018-07-03 one at 15:22. Counting them, those sessions hold 250-270 bars rather than
# the 210 a clean early close would give.
#
# So the rule counts the *afternoon*: a full session prints in nearly all 180 minutes between 13:00
# and 16:00, a half day in a few dozen. Anything in between is neither, and `suspicious` says so
# rather than guessing.
BUSY_AFTERNOON = 90  # half the afternoon's minutes; a real half day is far below it


def sessions(index: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per trading day: when it opened, when its last bar opened, how many bars it held."""
    local = pd.DatetimeIndex(index).tz_convert(EXCHANGE)
    by_day = pd.Series(local, index=local.date).groupby(level=0)
    afternoon = pd.Series(local[local.time >= EARLY_CLOSE], index=local[local.time >= EARLY_CLOSE].date)
    out = pd.DataFrame(
        {
            "open": by_day.min().dt.time,
            "close": by_day.max().dt.time,
            "bars": by_day.size(),
            "utc_open": pd.Series(pd.DatetimeIndex(index), index=local.date).groupby(level=0).min().dt.time,
            "afternoon": afternoon.groupby(level=0).size(),
        }
    )
    out.index = pd.to_datetime(out.index)
    out["afternoon"] = out.afternoon.fillna(0).astype(int)
    return out.assign(half_day=out.afternoon < BUSY_AFTERNOON)


def consensus(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """One calendar from many symbols, by taking the median of each column across them.

    The union of indices is the wrong basis for a calendar and the store proves it: on a half day
    each symbol prints a handful of late, out-of-sequence trades, and twenty symbols' handfuls add
    up to a busy-looking afternoon. Measured on the built store, the union finds 8 half days in
    10.7 years where a single symbol finds 17 to 21 and the exchange had about 25.

    The median is the market's answer rather than any one name's: a session is a half day when most
    of the basket stopped trading at one, which is what a half day is. The union stays the right
    basis for *coverage* — which dates exist at all — and that is what `holidays` reads.
    """
    stacked = pd.concat(frames, keys=range(len(frames)))
    by_day = stacked.groupby(level=1)
    out = pd.DataFrame(
        {
            "open": by_day.open.min(),
            "close": by_day.close.max(),
            "bars": by_day.bars.median().astype(int),
            "utc_open": by_day.utc_open.min(),
            "afternoon": by_day.afternoon.median().astype(int),
        }
    )
    return out.assign(half_day=out.afternoon < BUSY_AFTERNOON)


def holidays(days: pd.DataFrame) -> pd.DatetimeIndex:
    """Weekdays inside the covered span on which the basket did not trade at all."""
    span = pd.date_range(days.index.min(), days.index.max(), freq="B")
    return span.difference(days.index)


def suspicious(days: pd.DataFrame, expected: int = FULL_SESSION, tolerance: float = 0.98) -> pd.DataFrame:
    """Sessions that are neither a full day nor a clean half day — and are worth a look.

    The tolerance is measured, not chosen: on the built store a full session holds 390 bars at the
    median and 388 at the 5th percentile, because the median symbol occasionally goes a minute with
    no print. Flagging every session under 390 marks 22% of the decade as broken and says nothing.
    At 98% — 382 bars — it marks 7 sessions in 10.7 years, and four of them are 2020-03-09, 03-12,
    03-16 and 03-18 at 376 bars: the COVID circuit breakers, a fourteen-minute market-wide halt.
    That is the point of the tolerance. It cannot tell a halt from a hole, and it is not supposed
    to; it narrows a decade to the handful of days where something actually happened.
    """
    full = (~days.half_day) & (days.bars < tolerance * expected)
    # A real half day holds its morning whole: 210 minutes from the bell to the early close. Fewer
    # than that is a session with a hole in it, whatever its afternoon looks like.
    # The same tolerance on the other branch, for the same reason: a half day whose morning holds
    # 208 of 210 minutes is two quiet minutes, not a hole, and without this it read as an anomaly.
    short = days.half_day & (days.bars - days.afternoon < tolerance * HALF_SESSION)
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
    assert days.afternoon.iloc[2] == 0 and days.afternoon.iloc[0] == AFTERNOON

    # The real shape of a half day in this store: the tape closed at 13:00 and then printed late,
    # sparsely, until the middle of the afternoon. Detected by a quiet afternoon and not by a last
    # timestamp, which is what the measurement forced — the naive rule called this a full session.
    late = day("2026-01-08", HALF_SESSION).append(
        pd.DatetimeIndex(
            [pd.Timestamp(f"2026-01-08 {t}", tz=EXCHANGE) for t in ("13:44", "14:10", "15:35")]
        ).tz_convert("UTC")
    )
    tape = sessions(late)
    assert tape.close.iloc[0] == time(15, 35), "the last print is deep into the afternoon"
    assert bool(tape.half_day.iloc[0]), "and it is still a half day, because the afternoon is empty"
    assert tape.afternoon.iloc[0] == 3 and tape.bars.iloc[0] == HALF_SESSION + 3
    assert suspicious(tape).empty, "its morning is whole, so it is not suspicious"

    # The consensus: one symbol printing all afternoon does not turn the basket's half day into a
    # full one, which is exactly what the union of indices did on the real store.
    busy = sessions(day("2026-01-08", FULL_SESSION))
    quiet = [sessions(late) for _ in range(3)]
    both = consensus(quiet + [busy])
    assert bool(both.half_day.iloc[0]), "three quiet names outvote one busy one"
    assert consensus([busy, busy, sessions(late)]).half_day.iloc[0] is not True, "and the reverse holds"

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


def audit() -> pd.DataFrame:
    """The calendar of the built store, and the three traps of §3 answered on real bars."""
    from alpacatrader.data.store import load, read_stamp
    from alpacatrader.universe import U1

    have = [s for s in U1.values() if read_stamp(s)]
    per_symbol = [sessions(load(symbol).index) for symbol in have]
    days = consensus(per_symbol)

    print(
        f"{len(have)}/{len(U1)} symbols · {len(days):,} sessions · "
        f"{days.index.min().date()} -> {days.index.max().date()}"
    )
    print(f"trap 1  DST: session opens at {sorted(set(days.open))} local, {sorted(set(days.utc_open))} UTC")
    print(f"        the UTC hour moved {len(dst_changes(days))} times; the local one never")
    half = days[days.half_day]
    print(
        f"trap 2  half days: {len(half)}, {len(half) / (len(days) / 252):.1f}/yr, "
        f"median {int(half.bars.median())} bars of which {int(half.afternoon.median())} after 13:00"
    )
    print(f"        their last print lands at {min(half.close)}-{max(half.close)}, well past the 13:00 close")
    full = days[~days.half_day]
    print(f"        full sessions: {int(full.bars.median())} bars, {int(full.afternoon.median())} after 13:00")
    odd = suspicious(days)
    print(f"        neither, at 98% of a session: {len(odd)} in {len(days) / 252:.1f} years")
    if len(odd):
        print("        " + ", ".join(f"{d.date()} ({b})" for d, b in odd.bars.items()))
    print(f"holidays: {len(holidays(days))} weekdays with no session, {len(holidays(days)) / (len(days) / 252):.1f}/yr")
    return days


if __name__ == "__main__":
    import sys

    if "--audit" in sys.argv:
        audit()
    else:
        _selfcheck()
        print("ok — holidays, half days and the DST move all read off the bars")
