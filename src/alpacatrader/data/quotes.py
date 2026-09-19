"""Measure 1b: the quoted spread of each candidate instrument, in cents.

§2 prices every instrument at a spread of exactly one penny and says so: that is a **lower bound**,
to be replaced by this measurement. At these prices the half spread is most of the cost, so a
candidate whose real spread is three cents is three times the instrument the table assumed.

**The rule, fixed before any output was looked at** (§4): inside an exposure already chosen by
`universe`, the winner is the instrument with the lowest cost per side by the formula of §2, using
the **time-weighted median spread inside RTH, excluding the first and last half hour**.

Why time-weighted and not a plain median over quotes: a symbol that reprices a thousand times in a
volatile minute and twice in a quiet one would otherwise be described almost entirely by its
volatile minutes. What a resting order actually meets is the spread that was *standing* the longest.

Why the opening and closing half hours are cut: both are a different market — the auction and its
tail carry a large share of the day's volume at spreads that are not what an intraday rotation
meets at 11:00. Including them would flatter a wide instrument and penalise a tight one.

**Sampled, not exhaustive, and the sampling is the honest part.** A full RTH of NBBO for thirty
instruments is tens of millions of rows for a number that is a median. This takes short windows
spread evenly across several sessions, weights each quote by how long it stood, and takes the median
over the pooled sample. `SESSIONS` and `PER_SESSION` are what buys precision; both are arguments.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.requests import StockQuotesRequest

from alpacatrader.data.candles import EXCHANGE, client

# Half an hour inside each end of the session, per the decision rule.
SAMPLE_OPEN, SAMPLE_CLOSE = time(10, 0), time(15, 30)
PER_SESSION = 8  # windows per session, evenly spaced across the sampled stretch
SECONDS = 30  # length of one window
# A spread wider than this share of the price is not a quote an intraday book meets — it is a
# halt, an auction imbalance or a stale one-sided book. Dropped rather than winsorised, because a
# median is robust to the count of what is removed but not to a fabricated value.
MAX_SPREAD_FRACTION = 0.05


def windows(sessions: list[str], per_session: int = PER_SESSION, seconds: int = SECONDS):
    """`(start, end)` pairs in UTC, evenly spaced inside the sampled stretch of each session.

    Built in exchange time and converted, so the two DST changes a year move the UTC times and
    never the session times — the same reason `data.candles.regular_hours` compares where it does.
    """
    out = []
    for day in sessions:
        first = pd.Timestamp(f"{day} {SAMPLE_OPEN}", tz=EXCHANGE)
        last = pd.Timestamp(f"{day} {SAMPLE_CLOSE}", tz=EXCHANGE) - pd.Timedelta(seconds=seconds)
        for start in pd.date_range(first, last, periods=per_session):
            out.append((start.tz_convert("UTC"), (start + pd.Timedelta(seconds=seconds)).tz_convert("UTC")))
    return out


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The value at which half the *weight* lies below.

    Not `np.median` on repeated values: the weights here are durations in seconds and are continuous,
    so there is nothing to repeat. Ties and a single sample both fall out of the same search.
    """
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    return float(values[np.searchsorted(np.cumsum(weights), weights.sum() / 2.0)])


def sample(symbols: list[str], sessions: list[str], per_session: int = PER_SESSION, seconds: int = SECONDS):
    """Every sampled quote of every symbol, with the seconds each one stood.

    One request per window for the whole list, not one per symbol: the endpoint takes a list, and
    thirty-four separate calls per window would be thirty-four times the latency for the same rows.

    The last quote of a window is weighted to the window's end and not dropped. In a quiet name that
    quote is most of the window, and dropping it would delete exactly the wide, stale books this
    measurement exists to catch.
    """
    pieces = []
    for start, end in windows(sessions, per_session, seconds):
        df = (
            client()
            .get_stock_quotes(StockQuotesRequest(symbol_or_symbols=symbols, start=start, end=end, feed=DataFeed.SIP))
            .df
        )
        if df.empty:
            continue
        df = df[["bid_price", "ask_price"]].reset_index()
        df["held"] = df.groupby("symbol").timestamp.diff().shift(-1).dt.total_seconds()
        df["held"] = df.held.fillna((end - df.timestamp).dt.total_seconds()).clip(lower=0)
        pieces.append(df)
    if not pieces:
        return pd.DataFrame(columns=["symbol", "timestamp", "bid_price", "ask_price", "held"])
    out = pd.concat(pieces, ignore_index=True)
    out["spread"] = out.ask_price - out.bid_price
    out["mid"] = (out.ask_price + out.bid_price) / 2
    keep = (out.spread > 0) & (out.mid > 0) & (out.spread < MAX_SPREAD_FRACTION * out.mid) & (out.held > 0)
    return out[keep]


def measure(symbols: list[str], sessions: list[str], **kwargs) -> pd.DataFrame:
    """Time-weighted median spread and mid price per symbol, plus how much weight backs each."""
    rows = sample(symbols, sessions, **kwargs)
    out = {}
    for symbol, g in rows.groupby("symbol"):
        out[symbol] = {
            "spread": weighted_median(g.spread.to_numpy(), g.held.to_numpy()),
            "price": weighted_median(g.mid.to_numpy(), g.held.to_numpy()),
            "quotes": len(g),
            "seconds": round(float(g.held.sum()), 1),
        }
    return pd.DataFrame(out).T.sort_index()


def _selfcheck() -> None:
    """The weighting and the windows, on values built here — the fetch needs the network."""
    # The whole point of weighting: a value that stood for one second and repriced a hundred times
    # must not outvote one that stood for the rest of the window.
    noisy = np.array([0.05] * 100 + [0.01])
    held = np.array([0.01] * 100 + [29.0])
    assert weighted_median(noisy, held) == 0.01, "the standing quote wins, not the busy one"
    assert float(np.median(noisy)) == 0.05, "which is exactly where a plain median would land"
    assert weighted_median(np.array([0.02]), np.array([30.0])) == 0.02
    # Equal weights degenerate to the ordinary median, so the estimator is not a different thing.
    even = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert weighted_median(even, np.ones(5)) == 3.0

    # The windows sit inside the sampled stretch, never on the excluded half hours, and stay there
    # across the DST change — January is UTC-5 and July UTC-4, and the session times do not move.
    for day, offset in (("2026-01-15", 5), ("2026-07-15", 4)):
        w = windows([day], per_session=4, seconds=SECONDS)
        assert len(w) == 4
        local = [(s.tz_convert(EXCHANGE), e.tz_convert(EXCHANGE)) for s, e in w]
        assert local[0][0].time() == SAMPLE_OPEN, "the first window opens half an hour after the bell"
        assert local[-1][1].time() == SAMPLE_CLOSE, "and the last one ends half an hour before it"
        assert all(SAMPLE_OPEN <= s.time() and e.time() <= SAMPLE_CLOSE for s, e in local)
        assert w[0][0].hour == SAMPLE_OPEN.hour + offset, "the UTC hour moves with DST, the session does not"
        assert all((e - s).total_seconds() == SECONDS for s, e in w)
        assert w[1][0] > w[0][1], "windows do not overlap"

    assert len(windows(["2026-01-15", "2026-01-16"], per_session=3)) == 6


SESSIONS = ["2026-09-16", "2026-09-17", "2026-09-18"]


def run(sessions: list[str] = SESSIONS) -> pd.DataFrame:
    """Take measure 1b again and print the decision. `universe.U1` is the decision itself."""
    from alpacatrader.costs import per_side_bp
    from alpacatrader.universe import E1, EXPOSURES, U1

    candidates = sorted({i for e in E1 for i in EXPOSURES[e]})
    m = measure(candidates, sessions)
    rows = [
        {
            "exposure": e,
            "symbol": i,
            "price": round(m.price[i], 2),
            "cents": round(m.spread[i] * 100),
            "bp": round(per_side_bp(m.price[i], m.spread[i]), 3),
            "quotes": int(m.quotes[i]),
        }
        for e in E1
        for i in EXPOSURES[e]
        if i in m.index
    ]
    out = pd.DataFrame(rows)
    won = out.loc[out.groupby("exposure").bp.idxmin()].set_index("exposure").symbol
    print(
        out.assign(win=[" <-" if won.get(r.exposure) == r.symbol else "" for r in out.itertuples()]).to_string(
            index=False
        )
    )
    if dict(won.loc[E1]) != U1:
        print("\nWARNING: the spreads have moved the winner of an exposure. U1 is a decision —")
        print("re-take it deliberately, with the date, rather than editing the constant.")
    return out


if __name__ == "__main__":
    import sys

    if "--run" in sys.argv:
        run()
    else:
        _selfcheck()
        print("ok — the standing quote outweighs the busy one")
