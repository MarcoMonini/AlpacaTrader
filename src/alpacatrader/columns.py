"""The causal candidate columns, on session time.

Ported from the previous project's `legs.exhaustion` and `factor.composite`, which §6 marks
*da rimisurare*: all of them are causal and scale-free by construction and apply to an equity
candle exactly as to a crypto one, but what they *measure* changes and the windows are new.

**Every window restarts at the open.** Two kinds, and they are not interchangeable:

* A **rolling** window of `n` bars is correct globally for any bar at least `n - 1` into its
  session, because those `n` rows are all that session's. Only the first `n - 1` bars of each day
  need masking, and `usable` already excludes far more than that. Computed globally and masked,
  which is fast and exact.
* An **exponential** window is recursive and never forgets. At span 15 a bar 30 into the session
  still carries 13% of yesterday, so `(14/15)^30` of last night's close leaks into this morning's
  RSI. These are grouped by session and restarted, which costs time and is the only correct way.

Mixing the two up is the failure §7 is written around, so `_rolling` and `_ewm` are separate
functions and neither is a wrapper for the other.

**The intraday U, and why the cross-section absorbs most of it.** §6 warns that the day's shape
dominates six volume columns and that a model handed the hour scores high against any label with
intraday structure. Cross-sectionally that warning is weaker than it reads: the hour is the *same*
number for every symbol at an instant, so a pure time-of-day column has no cross-sectional
dispersion at all and cannot produce a Rank IC. What survives is the part of the U that differs
*between* symbols, and `volume_climax` is where it lives. `seasonal` is the remedy §6 calls (i) —
each symbol's volume against its own history at the same minute of the session — built causally
from past sessions only, so it needs no train/test split to be honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXHAUSTION = (
    "divergence",
    "volume_climax",
    "volume_climax_seasonal",
    "rejection",
    "streak",
    "deceleration",
    "stretch",
)

# The background window three of the columns measure against — volatility for `stretch` and
# `deceleration`, volume for `volume_climax`. The previous project used `4 * window`, which on a
# market that never closes is 24 hours of context and costs nothing. Here it does not fit: at
# `4 * 15 = 60` bars the statistic first exists at bar 59, while the usable stretch starts at bar
# 30, so 29 of the 85 usable bars come back empty and a 70-bar half day contributes nothing at all.
# Measured before the change: 58.6% coverage on `stretch` and `deceleration`, 59.7% on
# `volume_climax`, against 94% on the columns that do fit.
#
# `2 * window` is the largest multiple that fits, and it is not a compromise but the same rule that
# sized N and W: a window has to close inside the session that holds it. 30 bars is 90 minutes,
# twice the measured leg, and the statistic is then available exactly where the sample begins.
BACKGROUND = 2


def _mask(wide: pd.DataFrame, keep) -> pd.DataFrame:
    """Blank whole rows of a wide frame. `DataFrame.where` does not broadcast a column vector."""
    return wide.where(np.broadcast_to(np.asarray(keep)[:, None], wide.shape))


def _same_session(session: pd.Series) -> np.ndarray:
    """Rows whose previous row belongs to the same session — where a `diff` is a real difference."""
    return (session == session.shift()).to_numpy()


def _rolling(wide: pd.DataFrame, bar: pd.Series, window: int, how: str = "mean") -> pd.DataFrame:
    """A rolling statistic that is exact from bar `window - 1` of each session and NaN before it."""
    return _mask(getattr(wide.rolling(window), how)(), bar >= window - 1)


def _ewm(wide: pd.DataFrame, session: pd.Series, **kwargs) -> pd.DataFrame:
    """An exponential statistic restarted at every open — the recursion may not remember yesterday."""
    return wide.groupby(session.to_numpy()).ewm(**kwargs).mean().reset_index(level=0, drop=True)


def rsi(close: pd.DataFrame, session: pd.Series, window: int) -> pd.DataFrame:
    """Wilder's RSI in [0, 1], restarted at every open.

    Written out rather than taken from `ta`: it is three lines of `ewm(alpha=1/n, adjust=False)`,
    and the library's version cannot be restarted per session without being called once per day per
    symbol. The previous project keeps `ta` as the definition for PSAR, where the recursion is
    path-dependent and bit-equality matters; nothing here depends on that.
    """
    delta = _mask(close.diff(), _same_session(session))
    up = _ewm(delta.clip(lower=0), session, alpha=1 / window, adjust=False)
    down = _ewm((-delta).clip(lower=0), session, alpha=1 / window, adjust=False)
    return up / (up + down).replace(0, np.nan)


def seasonal(volume: pd.DataFrame, session: pd.Series, bar: pd.Series) -> pd.DataFrame:
    """Volume against the same symbol's own history at the same minute of the session.

    §6's remedy (i), and the only one of the two that is not degenerate cross-sectionally. Built on
    an **expanding** median over past sessions and shifted, so no row reads its own day or any later
    one — causal by construction, which is what lets it skip the train-only rule rather than break
    it. The first sessions have no history and come back NaN, which `usable` and the warm-up of the
    folds both already absorb.

    Pivoted to sessions-by-bars rather than grouped and applied. A `groupby(...).apply(...)` returns
    its rows in *group* order, so reading the result back as an array lines every value up against
    the wrong row — silently, and in a way no shape check would catch.
    """
    key = pd.MultiIndex.from_arrays([session.to_numpy(), bar.to_numpy()])
    out = {}
    for name in volume.columns:
        # rows are sessions, columns are the minute of the session: `expanding` then walks sessions.
        grid = volume[name].groupby([session.to_numpy(), bar.to_numpy()]).first().unstack()
        past = grid.expanding().median().shift()
        out[name] = past.stack(future_stack=True).reindex(key).to_numpy()
    background = pd.DataFrame(out, index=volume.index)
    return np.log(volume.replace(0, np.nan) / background.replace(0, np.nan))


def exhaustion(p: dict, window: int) -> dict[str, pd.DataFrame]:
    """Six ways a move runs out, all causal, one wide frame each.

    `divergence` is the classic one as a difference of ranks: where the close sits in its window
    against where the RSI sits in its own, so a new high on a weaker oscillator is negative.
    `volume_climax` is volume against its own background, signed by where the price is in its range
    — a spike at an extreme, which is what capitulation looks like. `rejection` is the wick on the
    far side of the move, counted only where it contradicts. `streak` is the run of same-signed
    closes, squashed. `deceleration` is the move losing speed against its own direction. `stretch`
    is how far the close has left its own average, in units of what the volatility would produce.
    """
    close, high, low, volume, opens = p["close"], p["high"], p["low"], p["volume"], p["open"]
    session, bar = p["session"], p["bar"]
    span = (high - low).replace(0, np.nan)
    inside = _same_session(session)
    price_rank = _mask(close.rolling(window).rank(pct=True), bar >= window - 1)
    position = 2 * price_rank - 1
    rsi_rank = _mask(rsi(close, session, window).rolling(window).rank(pct=True), bar >= window - 1)

    upper = (high - np.maximum(close, opens)) / span
    lower = (np.minimum(close, opens) - low) / span
    step = np.sign(_mask(close.diff(), inside)).fillna(0.0)
    # The run of same-signed closes, restarted at every open: a new run starts where the sign
    # changes **or** where the session does, so the first bar of a day never inherits yesterday's
    # momentum. A streak may not survive a night.
    opened = pd.DataFrame(np.broadcast_to((~inside)[:, None], step.shape), index=step.index, columns=step.columns)
    group = ((step != step.shift()) | opened).cumsum()
    run = group.apply(lambda col: col.groupby(col).cumcount() + 1)

    ema = _ewm(close, session, span=window, adjust=False)
    slope = _mask(np.log(ema).diff(), inside)
    sigma = _rolling(_mask(np.log(close).diff(), inside), bar, BACKGROUND * window, "std")
    background = _rolling(volume, bar, BACKGROUND * window, "mean")
    spread = _rolling(volume, bar, BACKGROUND * window, "std")

    return {
        "divergence": price_rank - rsi_rank,
        # Two backgrounds for the same statistic, because §6 says to compare the remedies rather
        # than pick one: within the session against its own recent bars, and across sessions against
        # this symbol's own history at the same minute of the day. The first is blind to the daily U
        # and the second is built out of it.
        "volume_climax": (volume - background) / spread.replace(0, np.nan) * position,
        "volume_climax_seasonal": seasonal(volume, session, bar) * position,
        "rejection": lower * np.maximum(-position, 0) - upper * np.maximum(position, 0),
        "streak": step * np.tanh(run / window),
        "deceleration": -np.sign(slope) * slope.diff() / sigma.replace(0, np.nan),
        "stretch": (np.log(close) - np.log(ema)) / sigma.replace(0, np.nan),
    }


def cross_rank(wide: pd.DataFrame) -> pd.DataFrame:
    """Centred percentile inside each row.

    Centred on the row mean rather than on a flat 0.5, so a cross-section that thins does not carry
    an offset made of nothing but its own count.
    """
    ranked = wide.rank(axis=1, pct=True)
    return ranked.sub(ranked.mean(axis=1), axis=0)


def composite(p: dict, window: int) -> pd.DataFrame:
    """`rank(log dollar volume) − rank(realized volatility)` over `window` bars. No parameters.

    The two columns the previous project's selection kept, with the signs of the low-volatility
    anomaly and the size effect, equal weight because the fitted alternative sat inside the fold
    spread. It is the thing every fitted model there failed to beat, which is why it is the first
    thing measured here and not the last.
    """
    session = p["session"]
    ret = _mask(np.log(p["close"]).diff(), _same_session(session))
    volatility = _rolling(ret, p["bar"], window, "std")
    size = np.log(_rolling(p["close"] * p["volume"], p["bar"], window, "mean").replace(0, np.nan))
    return cross_rank(size) - cross_rank(volatility)


def _selfcheck() -> None:
    """The two window kinds and the session restart — no store and no network."""
    per, days = 40, 3
    session = pd.Series(np.repeat(np.arange(days), per))
    bar = pd.Series(np.tile(np.arange(per), days))
    rng = np.random.default_rng(0)
    close = pd.DataFrame(np.exp(np.cumsum(rng.normal(scale=1e-3, size=(per * days, 3)), axis=0)), columns=list("abc"))

    # A rolling window is NaN for the first n-1 bars of every session, and exact after.
    got = _rolling(close, bar, 10)
    assert got[bar.to_numpy() < 9].isna().all().all() and got[bar.to_numpy() >= 9].notna().all().all()
    # Exact means equal to the same window computed on that session alone.
    one = close.iloc[per : 2 * per]
    assert np.allclose(got.iloc[per + 9].to_numpy(), one.rolling(10).mean().iloc[9].to_numpy())

    # An exponential window restarts: the first bar of a session is its own value, not yesterday's.
    smooth = _ewm(close, session, span=10, adjust=False)
    assert np.allclose(smooth.iloc[per].to_numpy(), close.iloc[per].to_numpy()), "the recursion restarts"
    # And a global ewm would not — which is the whole reason this function exists.
    assert not np.allclose(close.ewm(span=10, adjust=False).mean().iloc[per].to_numpy(), close.iloc[per].to_numpy())

    # RSI stays in [0, 1], restarts, and reads 1 on a series that only rises.
    strength = rsi(close, session, 14)
    valid = strength.dropna()
    assert ((valid >= 0) & (valid <= 1)).all().all()
    rising = pd.DataFrame({"a": np.arange(per * days, dtype=float)})
    assert np.isclose(rsi(rising, session, 14).iloc[per + 5, 0], 1.0), "only gains is an RSI of one"

    # `seasonal` never reads its own session or a later one: the first session is all NaN.
    volume = pd.DataFrame(rng.lognormal(size=(per * days, 3)), columns=list("abc"))
    seas = seasonal(volume, session, bar)
    assert seas.iloc[:per].isna().all().all(), "the first session has no past to be measured against"
    assert seas.iloc[per:].notna().any().any()
    # A symbol whose volume repeats exactly reads zero against its own history.
    steady = pd.DataFrame({"a": np.tile(np.arange(1.0, per + 1), days)})
    assert np.allclose(seasonal(steady, session, bar).iloc[2 * per :, 0].to_numpy(), 0.0)

    # `cross_rank` centres on the row, so a row sums to zero whatever its width.
    assert np.allclose(cross_rank(close).sum(axis=1).to_numpy(), 0.0, atol=1e-12)
    thin = close.copy()
    thin.iloc[0, 1:] = np.nan
    assert np.isclose(cross_rank(thin).iloc[0].sum(), 0.0, atol=1e-12)

    # The six columns come out, finite or NaN and never an infinity — `dropna` does not see one.
    p = {
        "open": close,
        "high": close * 1.001,
        "low": close * 0.999,
        "close": close,
        "volume": volume,
        "session": session,
        "bar": bar,
    }
    out = exhaustion(p, window=5)
    assert set(out) == set(EXHAUSTION)
    for name, frame in out.items():
        assert not np.isinf(frame.to_numpy()).any(), f"{name} carries an infinity"
        assert frame.shape == close.shape
    assert composite(p, window=5).shape == close.shape


if __name__ == "__main__":
    _selfcheck()
    print("ok — rolling is masked, exponential is restarted, and neither remembers yesterday")
