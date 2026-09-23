"""The cross-section of U1 on session time: closes, forward returns, folds.

What every measure from here on reads. A rank is a statement about a symbol *against the others
trading at that instant*, so the unit is the panel and not the series, and §9's first instruction
about any prediction file is to check its breadth before believing a single number taken on it.

**The forward return never crosses a close.** In the intraday-only regime a position is flat at
16:00, so a return reaching past the bell is one no rule could have collected — it is the overnight
gap, which §7 removes from the label and §5 calls the largest source of untradable content there
is. A row whose horizon would reach past its session's last bar has no forward return and is
dropped, rather than truncated: a truncated horizon makes the metric a mixture of horizons, and
the average of that is not the average of anything.

Measured cost of the rule, on the 85 usable bars of a 130-bar session:

    h (bars)    minutes    rows kept    % of usable
        5          15       227,636       100.0
       15          45       227,636       100.0
       30          90       187,241        82.3
       60         180       106,871         46.9

Up to 15 bars it is free, because the extrema window already cuts the same tail. `HORIZONS` is
anchored on the measured leg — 17 bars at W=15 — so the three readings are a third of a leg, one
leg, and two: the same short / label-scale / long shape the previous project read at 6, 24 and 72.

**Market-neutral, always.** The cross-sectional mean is removed at every instant, which is what
makes the number a statement about *selection* rather than about the basket having gone up. On a
rising basket, quoting a raw return is how a long-biased rule gets mistaken for a skilled one —
the mirror of the mistake the previous project made on a falling one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpacatrader.session import EXTREMA, STEPS, frame, half_days
from alpacatrader.universe import U1

HORIZONS = (5, 15, 30)  # bars of 3m: a third of a leg, a leg, two legs
FOLDS = 4


FIELDS = ("open", "high", "low", "close", "volume")


def build(symbols: list[str] | None = None) -> dict:
    """OHLCV as wide frames, plus session, bar and the usable mask, on one grid every symbol shares.

    Aligned on the timestamp and then renumbered: a symbol missing a bucket leaves a NaN in a row
    rather than shifting its own bar count, which is what would happen if each symbol carried its
    own index. `bar` is counted on the shared grid for the same reason.
    """
    names = list(symbols or list(U1.values()))
    # The half-day calendar is read once and handed to every symbol. `frame` would otherwise
    # rebuild it per call, and that means re-reading all twenty Parquet files twenty times.
    early = half_days(names)
    each = {symbol: frame(symbol, early=early).set_index("timestamp") for symbol in names}
    wide = {f: pd.DataFrame({s: b[f] for s, b in each.items()}).sort_index() for f in FIELDS}
    close = wide["close"]
    day = pd.Series(close.index.tz_convert("America/New_York").date, index=close.index)
    session = pd.Series(day.factorize()[0], index=close.index)
    bar = session.groupby(session).cumcount()
    size = session.map(session.value_counts())
    keep = (bar >= max(2 * STEPS, EXTREMA)) & (bar < size - EXTREMA)
    return {**wide, "session": session, "bar": bar, "size": size, "usable": keep}


def forward(p: dict, horizon: int) -> pd.DataFrame:
    """Market-neutral log return over the next `horizon` bars, NaN where it would cross a close."""
    close, session, bar, size = p["close"], p["session"], p["bar"], p["size"]
    ahead = np.log(close.shift(-horizon)) - np.log(close)
    inside = (bar + horizon < size) & session.eq(session.shift(-horizon))
    ahead = ahead.where(inside, np.nan)
    return ahead.sub(ahead.mean(axis=1), axis=0)


def folds(p: dict, n: int = FOLDS) -> list[pd.Series]:
    """`n` expanding walk-forward test masks, cut on session boundaries.

    Cut on sessions and not on rows so a fold never begins mid-day, and expanding rather than
    rolling because that is the protocol every comparison in the previous project was made on. With
    10.7 years the four cuts finally fall in four different regimes, which §3 calls the largest
    single gain of the change of asset class.
    """
    session = p["session"]
    last = int(session.max())
    edges = [int(last * (i + 1) / (n + 1)) for i in range(n + 1)]
    return [session.between(lo + 1, hi) for lo, hi in zip(edges[:-1], edges[1:])]


def _selfcheck() -> None:
    """The forward rule, on a panel built here — the store needs no network but does need files."""
    per, n_sessions = 20, 3
    days = [
        pd.date_range(f"2026-01-0{5 + s} 09:30", periods=per, freq="3min", tz="America/New_York")
        for s in range(n_sessions)
    ]
    index = pd.DatetimeIndex(np.concatenate(days)).tz_convert("UTC")
    rng = np.random.default_rng(0)
    walk = np.exp(np.cumsum(rng.normal(scale=1e-3, size=(len(index), 4)), axis=0))
    close = pd.DataFrame(walk, index=index, columns=list("abcd"))
    day = pd.Series(close.index.tz_convert("America/New_York").date, index=close.index)
    session = pd.Series(day.factorize()[0], index=close.index)
    bar = session.groupby(session).cumcount()
    p = {"close": close, "session": session, "bar": bar, "size": session.map(session.value_counts())}

    f = forward(p, 5)
    # The last five bars of every session have no five-bar future inside it.
    assert f[bar.to_numpy() >= per - 5].isna().all().all(), "a forward return may not cross a close"
    assert f[bar.to_numpy() < per - 5].notna().all().all(), "and every earlier bar has one"
    assert len(f.dropna(how="all")) == n_sessions * (per - 5)

    # Market-neutral: every row sums to zero, which is what makes it about selection.
    assert np.allclose(f.dropna(how="all").sum(axis=1), 0.0, atol=1e-12)
    # And it is the raw return minus the row mean, not something else.
    raw = np.log(close.shift(-5)) - np.log(close)
    first = raw.iloc[0]
    assert np.allclose(f.iloc[0].to_numpy(), (first - first.mean()).to_numpy())

    # A horizon as long as the session leaves nothing, and says so rather than inventing a row.
    assert forward(p, per).isna().all().all()
    # A longer horizon is a subset of a shorter one's rows — never the reverse.
    short, long = forward(p, 3).notna().any(axis=1), forward(p, 8).notna().any(axis=1)
    assert (long <= short).all(), "every row with an 8-bar future has a 3-bar one"

    # Folds partition the sessions, expand, and never split a session in two.
    masks = folds(p, n=2)
    assert len(masks) == 2
    for mask in masks:
        touched = session[mask].unique()
        for s in touched:
            assert mask[session == s].all(), "a fold takes whole sessions or none"
    assert not (masks[0] & masks[1]).any(), "the test slices do not overlap"


if __name__ == "__main__":
    _selfcheck()
    print("ok — no forward return crosses a close, and every row is market-neutral")
