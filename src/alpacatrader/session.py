"""Measure 4: session time — the bar index that replaces the clock.

The rule the previous project depended on — "every frame is indexed by the open of its bar, so a bar
`b` on timeframe `tf` closes at `b + tf`" — assumes a bar duration always passes between adjacent
bars. On equity RTH, 17.5 hours pass between the 15:55 bar and the 09:30 one, and 65 over a weekend.
§7 lists what that breaks: a forward fill carries last night's branch into this morning's open, the
clock sampling `index.floor(step) == index` invents timestamps at 03:00, ATR reads the opening gap
as a bar's range, and a centred pivot pairs yesterday afternoon's high with this morning's.

**The correction is an index, not a patch.** Sessions are concatenated and numbered with an integer
that never jumps; the timestamp survives as a column, used only for the temporal split, the join
between branches and the corporate-action calendar. Everything downstream is already positional —
features run on N bars, `find_pivots` on `order` bars, the tensor on N steps, purging on row indices
— so nothing below this line has to learn about time.

**The derived rule, which is what the truncation test enforces:** no rolling window and no forward
fill crosses a close. Truncate the series at a close and no column of the following session may
change.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from alpacatrader.calendar import EARLY_CLOSE, consensus, sessions
from alpacatrader.data.candles import EXCHANGE
from alpacatrader.data.store import load

# The grid, fixed 2026-09-19 on the built store. 1m is disqualified as a base: median breadth 18.94
# and all twenty symbols present at only 33.1% of instants, against 19.74 and 80.2% at 3m — UUP
# covers 67.7% of 1m bars and 90.3% of 3m ones, and §9 refuses a file whose breadth is not ~20.
# 390 minutes divide into 130 three-minute bars exactly, and a half day's 210 into 70.
GRID = "3m"
PER_SESSION = 130

# The sequence window, fixed. 45 minutes of context, inside the one-hour activation ceiling. The
# warm-up is what sizes it and it is paid **every session**, because no window crosses a close:
# 2N bars of the 130 are gone before the first sample, so N trades context against sample count.
#
#   N     activation   warm-up   usable bars   % of session
#   10       30 min        20        110          84.6
#   12       36 min        24        106          81.5
#   15       45 min        30        100          76.9
#   16       48 min        32         98          75.4
#   20       60 min        40         90          69.2
STEPS = 15

# The extrema window of the pivots the leg runs between — **provisional**, and the only constant
# here that is not yet measured. §5 fixes the method: it is chosen by maximising the lag-penalised
# oracle P&L (measure 6), never by the naive criterion, whose argmax falls at 3 bars and is not
# stationary. 15 is where the grid below says to start, not where it ends.
#
# Measured 2026-09-19 on 3m bars of five symbols since 2024, with a centred rolling extremum
# standing in for `data.pivots.find_pivots` — the absolute numbers will move when the real detector
# lands, the ordering will not, since every row passes through the same approximation:
#
#   W    span    legs/session   median leg   R² of leg position vs time of day
#   10   30 min      6.4          36 min                 0.057
#   12   36 min      5.0          42 min                 0.082
#   15   45 min      3.5          51 min                 0.130
#   16   48 min      3.1          51 min                 0.149
#   20   60 min      2.0          63 min                 0.226
#
# 16 and 20 are excluded rather than carried forward. At W=20 a session holds two legs, so position
# along the leg *is* the hour of the day: 23% of the label's variance is the clock. §6 warns that a
# model handed the time of day scores high Rank IC against any label with intraday structure, and
# the six volume columns hand it over through the daily U. On a label §5 has already measured as
# retrospective, that would be two confounds multiplying.
#
# The cost runs the other way and is why this is measure 6's call and not a preference: trading
# every leg at the basket's median 0.730 bp per side costs 23.5% a year at W=10 and 7.4% at W=20.
EXTREMA = 15


@lru_cache(maxsize=4)
def _half_days(names: tuple[str, ...]) -> pd.DatetimeIndex:
    from alpacatrader.data.store import path, read_stamp

    per = [sessions(pd.read_parquet(path(s)).index) for s in names if read_stamp(s)]
    days = consensus(per)
    return pd.DatetimeIndex(days.index[days.half_day])


def half_days(symbols: list[str] | None = None) -> pd.DatetimeIndex:
    """The dates the exchange closed at 13:00, by majority across the basket (see `calendar`).

    Cached on the symbol tuple: it reads every Parquet in the basket to take the majority, and a
    caller building a panel asks for it once per symbol unless something remembers the answer.
    """
    from alpacatrader.data.store import path, read_stamp
    from alpacatrader.universe import U1

    names = symbols or list(U1.values())
    per = [sessions(pd.read_parquet(path(s)).index) for s in names if read_stamp(s)]
    days = consensus(per)
    return pd.DatetimeIndex(days.index[days.half_day])


def trim_early(bars: pd.DataFrame, early: pd.DatetimeIndex) -> pd.DataFrame:
    """Drop what the tape printed after a half day's 13:00 close — see `frame`."""
    local = bars.index.tz_convert(EXCHANGE)
    shut = pd.DatetimeIndex(local.normalize().tz_localize(None)).isin(early) & (local.time >= EARLY_CLOSE)
    return bars[~shut]


def frame(symbol: str, grid: str = GRID, early: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """One symbol on session time: a row per bar, a session number, and the clock as a column.

    The index is a `RangeIndex` that never jumps. `session` counts sessions from zero and `bar`
    counts bars inside one, which is what makes "the first N bars of a session" expressible without
    a timestamp anywhere.

    **A half day is trimmed at its early close**, and that is not tidiness. The SIP tape keeps
    printing late and out-of-sequence trades for hours after the 13:00 bell — measured on IWM,
    2018-07-03 carries 48 three-minute bars between 13:00 and 15:21, contiguous enough to look like
    an ordinary afternoon. They are not a market: nothing could have been bought or sold at them.
    Left in, they would be 60% of that session's rows, feeding the features, the label and the P&L
    with prices no rule could ever have traded at.
    """
    bars = trim_early(load(symbol, grid), half_days() if early is None else early)
    day = pd.Series(bars.index.tz_convert(EXCHANGE).date, index=bars.index)
    out = bars.assign(timestamp=bars.index, session=day.factorize()[0])
    out["bar"] = out.groupby("session").cumcount()
    return out.reset_index(drop=True)


def rolling(df: pd.DataFrame, column: str, window: int, how: str = "mean") -> pd.Series:
    """A rolling statistic that restarts at every open.

    The one function every feature has to go through. A plain `df[column].rolling(window)` spans the
    close silently and produces a number whose inputs are two different days — the failure §7 is
    written around. Grouping by session makes the first `window - 1` bars of each day NaN, which is
    correct: there is no such window there, and a value would be an invention.
    """
    return getattr(df.groupby("session")[column].rolling(window), how)().reset_index(level=0, drop=True)


def usable(df: pd.DataFrame, steps: int = STEPS, extrema: int = EXTREMA) -> pd.Series:
    """Rows that have both a full sequence behind them and a complete leg around them.

    Two windows eat the session from opposite ends. The head loses `2 * steps` bars — `steps` of
    sequence plus the `steps` the first step looks back over — and both ends lose `extrema`, because
    a centred pivot window cannot be closed at a boundary it may not cross. What is left is the
    sample, and on a 130-bar session at N=W=15 that is 85 bars, 65% of it.
    """
    size = df.groupby("session").bar.transform("size")
    return (df.bar >= max(2 * steps, extrema)) & (df.bar < size - extrema)


def _selfcheck() -> None:
    """The truncation test, and the two windows — no store and no network."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(3):
        for b in range(PER_SESSION):
            rows.append({"session": s, "bar": b, "close": 100 + rng.normal()})
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.date_range("2026-01-05", periods=len(df), freq="3min", tz="UTC")

    # --- the rule: no rolling window crosses a close ----------------------------------------------
    got = rolling(df, "close", 10)
    first = df.groupby("session").head(9).index
    assert got.loc[first].isna().all(), "the first nine bars of every session have no ten-bar window"
    assert got.drop(first).notna().all(), "and every other bar has one"
    # The value at an open is built from that session only — a plain rolling would reach backwards.
    naive = df.close.rolling(10).mean()
    opens = df.index[df.bar == 0]
    assert naive.loc[opens[1:]].notna().all(), "which is exactly what the naive version would do"

    # --- the truncation test, which is the one §7 asks for ----------------------------------------
    # Cut the series at a close and recompute: nothing in the sessions that remain may move. If a
    # window reached across the boundary, the last session's tail would change when the next one
    # disappears — that is how the previous project tested its alignment, and it is the same test.
    cut = df[df.session < 2].reset_index(drop=True)
    full = rolling(df, "close", 10)[: len(cut)].to_numpy()
    assert np.allclose(rolling(cut, "close", 10).to_numpy(), full, equal_nan=True), "a close is a wall"
    # And the other direction: dropping the *first* session must not move the last one either.
    tail = df[df.session > 0].reset_index(drop=True)
    assert np.allclose(
        rolling(tail, "close", 10).to_numpy(), rolling(df, "close", 10)[PER_SESSION:].to_numpy(), equal_nan=True
    )

    # --- the half-day trim -------------------------------------------------------------------------
    # Two days of 3m bars: a full one, and a half day whose tape keeps printing until mid-afternoon,
    # which is what the store really holds. Only the late prints go, and only on the declared date.
    full = pd.date_range("2026-01-07 09:30", periods=PER_SESSION, freq="3min", tz=EXCHANGE)
    early_day = pd.date_range("2026-01-08 09:30", periods=PER_SESSION, freq="3min", tz=EXCHANGE)
    bars = pd.DataFrame({"close": 1.0}, index=full.append(early_day).tz_convert("UTC"))
    kept = trim_early(bars, pd.DatetimeIndex([pd.Timestamp("2026-01-08")]))
    local = kept.index.tz_convert(EXCHANGE)
    assert (local.date == pd.Timestamp("2026-01-07").date()).sum() == PER_SESSION, "a full day is untouched"
    assert (local.date == pd.Timestamp("2026-01-08").date()).sum() == 70, "09:30 to 13:00 is 70 three-minute bars"
    assert local[local.date == pd.Timestamp("2026-01-08").date()].max().time() == pd.Timestamp("12:57").time()
    assert trim_early(bars, pd.DatetimeIndex([])).equals(bars), "no half day declared, nothing cut"

    # --- the two windows ---------------------------------------------------------------------------
    keep = usable(df, steps=15, extrema=15)
    per = keep.groupby(df.session).sum()
    assert (per == 85).all(), f"130 - max(30, 15) - 15 = 85 rows a session, got {per.tolist()}"
    assert df[keep].bar.min() == 30 and df[keep].bar.max() == PER_SESSION - 16
    # A shorter session — a half day — keeps fewer rows and needs no rule of its own.
    half = df[(df.session > 0) | (df.bar < 70)].reset_index(drop=True)
    assert int(usable(half).groupby(half.session).sum().iloc[0]) == 70 - 30 - 15
    # The extrema window alone decides the tail; the sequence window alone decides the head.
    assert df[usable(df, steps=15, extrema=5)].bar.max() == PER_SESSION - 6
    assert df[usable(df, steps=5, extrema=15)].bar.min() == 15, "when 2N < W, the pivot sets the head"


if __name__ == "__main__":
    _selfcheck()
    print("ok — a close is a wall, and truncation proves it")
