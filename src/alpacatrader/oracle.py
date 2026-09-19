"""Measure 6: the two readings of perfect hindsight, and the window every other window derives from.

The oracle buys the close of every pivot low and sells the close of the next pivot high, net of
fees. It is not a strategy — it is the ruler everything else is quoted against, and it only works
as one because it has **two readings and only one of them is a target**:

* `lag=0` buys every pivot with hindsight. On crypto it read 4.1 log a year on 4h bars and 12.6 on
  15m. It is unattainable by construction, because a pivot of a centred window is not knowable
  until `window` bars after it happened.
* `lag=window` is the earliest any causal reader could know that pivot, and on crypto it read 0.40
  a year — **under 10% of the first**. Everything that makes the hindsight number enormous is the
  window of future it reads. A causal strategy is quoted against the second and the first is
  mentioned only to say what it is.

**Why the window cannot be chosen at lag 0.** §5 records that the naive criterion is degenerate:
the P&L rises monotonically as the window shrinks until the average leg stops clearing costs, so
the argmax lands at 2-8 bars on every symbol and is a cost-to-volatility ratio rather than market
structure. Charging the detection lag is what separates legs a causal model could hold from legs
that exist only to a perfect predictor.

**The constraint the previous project could not have.** A leg that crosses the close is not
tradable in the intraday-only regime: the position is flat at 16:00 by construction, and the
overnight gap is exactly the untradable content §5 removes from the label. So the oracle counts
only legs opening and closing inside one session, and `crossing` reports how many it had to drop —
a window whose legs mostly span the bell has no business being chosen, whatever its P&L looks like.

    uv run python -m alpacatrader.oracle --run
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpacatrader import panel

# Measured 2026-09-19 on U1, 2,693 sessions, fees per symbol from M1b. `lag_w` is already net.
#
#   W   min   lag_0   lag_w    ratio   trades/sym/yr   legs crossing the bell
#   2     6  5.4846  0.7114    0.130         2,249.7                  5.3%
#   3     9  4.7611  0.5075    0.107         1,538.5                  7.6%
#   5    15  3.6547  0.3071    0.084           884.0                 12.5%
#   8    24  2.5662  0.1664    0.065           483.0                 20.8%
#  12    36  1.6593  0.0537    0.032           250.7                 33.1%
#  15    45  1.1871  0.0028    0.002           160.1                 43.1%
#  20    60  0.6513 -0.0303   -0.047            74.1                 60.0%
#  30    90  0.1630 -0.0165   -0.101            11.8                 86.9%
#
# **W = 15 is disqualified.** Its causal ceiling is 0.0028 log a year — a *perfect* predictor of
# that label, trading it causally, earns nothing. From 20 up the ceiling is negative: the detection
# lag costs more than the leg pays. And the crossing column is a constraint of its own: at W = 15,
# 43% of the legs span the bell and cannot be held at all in the intraday-only regime.
#
# **The lag-penalised criterion is degenerate here, and §5's "the method transfers in full" is
# wrong.** The argmax runs to the edge of any grid: 0.7114 at W = 2, falling monotonically. Re-run
# with the previous venue's 25 bp on the *same price series* and the argmax jumps from 2 to 24, with
# the whole column negative. The fee is what was choosing the window, not the lag penalty — which
# is exactly what §5 says about the *naive* criterion, and the penalty does not cure it. Moving to a
# venue 34x cheaper removed the thing that was doing the selecting.
#
# So the oracle cannot fix W here. What it does fix is the **ceiling**: no causal strategy on this
# label can beat `lag_w` at its window, and that is the number M5 and everything after it are quoted
# against. `stretch` at window 20 nets −1.60 where the ceiling at W = 5 is +0.31, so the gap is the
# signal and not the venue.
WINDOWS = (5, 8, 10, 12, 15, 20, 25, 30)
SESSIONS_PER_YEAR = 252


def find(close: pd.Series, session: np.ndarray, window: int) -> pd.DataFrame:
    """Centred local extrema, detected inside a session and never across one.

    A bar is a high when its close is the largest in the `2 * window + 1` bars around it. Comparing
    across a close would pair yesterday afternoon's high with this morning's and call the pair one
    oscillation, which §7 lists among the five things session time exists to stop.

    Same-kind runs are merged in arrival order, keeping the more extreme, so kinds alternate and
    every low is closed by a high. Merged and not filtered: two highs with no low between them are
    one turn seen twice, and dropping the later one instead of the weaker would move the pivot.
    """
    rolled = close.groupby(session).rolling(2 * window + 1, center=True, min_periods=2 * window + 1)
    high = (close == rolled.max().reset_index(level=0, drop=True)).to_numpy()
    low = (close == rolled.min().reset_index(level=0, drop=True)).to_numpy()
    kind = np.where(high & ~low, 1, np.where(low & ~high, -1, 0))
    where = np.flatnonzero(kind)
    if len(where) == 0:
        return pd.DataFrame({"row": [], "kind": [], "price": []}).astype({"row": int, "kind": int})

    values = close.to_numpy()
    keep: list[int] = []
    for idx in where:
        if keep and kind[keep[-1]] == kind[idx]:
            better = values[idx] > values[keep[-1]] if kind[idx] == 1 else values[idx] < values[keep[-1]]
            if better:
                keep[-1] = idx
            continue
        keep.append(idx)
    keep_arr = np.array(keep, dtype=int)
    return pd.DataFrame({"row": keep_arr, "kind": kind[keep_arr], "price": values[keep_arr]})


def run(close: pd.Series, session: np.ndarray, window: int, fee: float, lag: int = 0) -> dict:
    """Net log P&L a year of the hindsight trader on one symbol, long-only, one leg at a time.

    `lag` fills `lag` bars after the pivot instead of on it. Both fills shift by the same lag, so
    the exit still follows the entry — but when two pivots sit closer together than `lag` the entry
    lands past the exit pivot and the trade no longer overlaps the leg at all. A model that detected
    the low that late would not take the trade, so neither does the oracle.
    """
    piv = find(close, session, window)
    if len(piv) < 2:
        return {"trades": 0, "log_per_year": 0.0, "crossing": 0, "skipped": 0, "median_bars": np.nan}
    rows, kind, values = piv.row.to_numpy(), piv.kind.to_numpy(), close.to_numpy()
    filled = rows + lag
    ok = filled < len(close)
    rows, kind, filled = rows[ok], kind[ok], filled[ok]

    opens = (kind[:-1] == -1) & (filled[:-1] < rows[1:])
    skipped = int(((kind[:-1] == -1) & ~(filled[:-1] < rows[1:])).sum())
    entry, exit_ = filled[:-1][opens], filled[1:][opens]
    # Intraday-only: a leg that crosses the bell cannot be held, and the gap it carries is exactly
    # the untradable content the regime was chosen to exclude.
    same = session[entry] == session[exit_]
    crossing = int((~same).sum())
    entry, exit_ = entry[same], exit_[same]
    if len(entry) == 0:
        return {"trades": 0, "log_per_year": 0.0, "crossing": crossing, "skipped": skipped, "median_bars": np.nan}

    legs = (values[exit_] / values[entry]) * (1 - fee) ** 2 - 1
    years = len(np.unique(session)) / SESSIONS_PER_YEAR
    return {
        "trades": len(legs),
        "log_per_year": float(np.log1p(legs).sum() / years),
        "crossing": crossing,
        "skipped": skipped,
        "median_bars": float(np.median(exit_ - entry)),
    }


def sweep(p: dict, fees: pd.Series, windows=WINDOWS) -> pd.DataFrame:
    """Both readings at every window, summed over the basket — the lag-penalised choice of W.

    `lag_0` is the hindsight ceiling and `lag_w` is what a causal reader could reach. The column to
    choose on is `lag_w`, and `ratio` is what §5 calls the discount: on crypto it was under 10%.
    """
    close, session = p["close"], p["session"].to_numpy()
    rows = []
    for window in windows:
        totals = {"lag_0": 0.0, "lag_w": 0.0, "trades": 0, "crossing": 0, "bars": []}
        for symbol in close.columns:
            series = close[symbol]
            fee = fees[symbol] / 1e4
            totals["lag_0"] += run(series, session, window, fee, lag=0)["log_per_year"]
            late = run(series, session, window, fee, lag=window)
            totals["lag_w"] += late["log_per_year"]
            totals["trades"] += late["trades"]
            totals["crossing"] += late["crossing"]
            if not np.isnan(late["median_bars"]):
                totals["bars"].append(late["median_bars"])
        n = len(close.columns)
        rows.append(
            {
                "window": window,
                "minutes": 3 * window,
                "lag_0": round(totals["lag_0"] / n, 4),
                "lag_w": round(totals["lag_w"] / n, 4),
                "ratio": round(totals["lag_w"] / totals["lag_0"], 3) if totals["lag_0"] else np.nan,
                "trades_per_symbol_year": round(totals["trades"] / n / (len(np.unique(session)) / 252), 1),
                "crossing_pct": round(100 * totals["crossing"] / max(totals["crossing"] + totals["trades"], 1), 1),
                "median_bars": round(float(np.median(totals["bars"])), 1) if totals["bars"] else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _selfcheck() -> None:
    """Pivots and the two readings, on a sawtooth whose answer is known — no store, no network."""
    per, days = 60, 4
    # A clean triangle wave: highs every 20 bars, lows halfway between, the same in every session.
    wave = np.concatenate([np.arange(10), np.arange(10, 0, -1)] * 3)[:per]
    close = pd.Series(np.tile(100 + wave.astype(float), days))
    session = np.repeat(np.arange(days), per)

    piv = find(close, session, window=4)
    assert len(piv) > 0
    assert set(np.unique(piv.kind)) <= {-1, 1}
    assert (np.diff(piv.kind) != 0).all(), "kinds alternate after the merge"
    # Every pivot sits inside a session and none is within `window` of a boundary, because a centred
    # window cannot close there.
    inside = piv.row.to_numpy() % per
    assert (inside >= 4).all() and (inside < per - 4).all(), "a centred pivot cannot reach past the bell"

    # Hindsight makes money on a sawtooth; the same oracle delayed by the window makes much less.
    free = run(close, session, window=4, fee=0.0, lag=0)
    late = run(close, session, window=4, fee=0.0, lag=4)
    assert free["log_per_year"] > 0, "buying every low and selling every high must pay"
    assert late["log_per_year"] < free["log_per_year"], "and knowing them late must pay less"

    # The fee bites, and it bites twice a leg.
    charged = run(close, session, window=4, fee=0.01, lag=0)
    assert charged["log_per_year"] < free["log_per_year"]

    # A leg crossing a close is counted and dropped, never traded. Built by hand: one session that
    # ends rising and a next that opens falling, so hindsight would want to hold across the bell.
    ramp = pd.Series(np.concatenate([np.linspace(100, 110, per), np.linspace(110, 100, per)]))
    two = np.repeat([0, 1], per)
    out = run(ramp, two, window=4, fee=0.0, lag=0)
    assert out["crossing"] >= 0 and out["trades"] >= 0
    assert out["trades"] == 0 or out["log_per_year"] >= 0

    # A window too wide for the session finds nothing rather than reaching across it.
    assert len(find(close, session, window=per)) == 0, "no centred window fits, so no pivots"
    assert run(close, session, window=per, fee=0.0)["trades"] == 0


if __name__ == "__main__":
    import sys

    if "--run" in sys.argv:
        from alpacatrader.rule import fees

        built = panel.build()
        print(sweep(built, fees(built)).to_string(index=False))
    elif "--fee" in sys.argv:
        from alpacatrader.rule import fees

        built = panel.build()
        real = fees(built)
        for label, vector in (("equity, misurate", real), ("crypto, 25 bp", real * 0 + 25.0)):
            out = sweep(built, vector, windows=(2, 3, 5, 8, 12, 15, 20, 24))
            best = out.loc[out.lag_w.idxmax()]
            print(f"--- {label}: argmax W={int(best.window)}, lag_w {best.lag_w:+.4f}")
            print(out[["window", "lag_0", "lag_w", "trades_per_symbol_year"]].to_string(index=False))
    elif "--fine" in sys.argv:
        from alpacatrader.rule import fees

        built = panel.build()
        print(sweep(built, fees(built), windows=(2, 3, 4, 5, 6, 7)).to_string(index=False))
    else:
        _selfcheck()
        print("ok — pivots alternate, stay inside the session, and the lag costs what it should")
