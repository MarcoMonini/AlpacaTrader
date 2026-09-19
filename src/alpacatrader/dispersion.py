"""Measure 3.5: is there enough cross-sectional dispersion for the label to mean anything?

The label is an ordering **inside the instant**, divided by the dispersion of that instant. If
`sd_t` is small, the label is noise divided by a small number, and nothing downstream can repair
that — not a better model, not a better rule. Twenty US sector and asset-class ETFs share a market
factor that twenty altcoins did not, and **at intraday frequency they may share it more**, which is
why `universe`'s daily correlation check cannot answer this and this module exists.

The spec places it immediately after the store is built. It is taken before instead, because it
does not need the store: a sample of sessions from a few different regimes answers it at a fraction
of the cost, and an hour that can invalidate the label belongs in front of four hours that assume it.

**Returns never cross a session boundary.** The first bar of a session carries the overnight gap,
which is a real return but not one any intraday rule can reach, and it is an order of magnitude
larger than a bar's. Leaving it in would inflate `sd_t` by exactly the quantity the intraday-only
regime of §7 exists to exclude, and the answer would be wrong in the flattering direction. This is
also the smallest piece of the session-time refactor (M4), taken early because this measure needs it.

**Compared against the crypto store with one implementation of the formula.** The previous project's
5m Parquet files are read directly — same `sd_t`, same resampling, same code path — because a
reference computed a second way eventually measures a second thing.

    uv run python -m alpacatrader.dispersion --run
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpacatrader.data.candles import EXCHANGE, bars

# The previous project's store, for the only comparison that puts this number in scale.
CRYPTO_STORE = Path("/Users/marcomonini/PycharmProjects/TradingVision/data")

# Three stretches in three regimes rather than one long recent one: dispersion is a property of a
# market, and 2026 alone would answer for 2026. Twenty sessions each is enough for a median.
PERIODS = {
    "2018 (vol)": ("2018-02-01", "2018-03-01"),
    "2022 (bear)": ("2022-06-01", "2022-07-01"),
    "2026 (recent)": ("2026-08-15", "2026-09-15"),
}
RULES = ["1min", "5min", "15min"]

# Measure 3.5, taken 2026-09-19 on SIP 1m bars of U1, resampled inside the session, against the
# previous project's 5m crypto store from 2023. Reproduce with `python -m alpacatrader.dispersion
# --run`. Taken before the store (M3) rather than after it, which the spec's order allows because
# sampled windows answer the question and an hour that can invalidate the label belongs in front.
#
#   set                bar   symbols   instants     sd_t
#   U1 2018 (vol)       1m        18       7391   0.000379
#   U1 2018 (vol)       5m        19       1463   0.000806
#   U1 2018 (vol)      15m        19        475   0.001295
#   U1 2022 (bear)      1m        20       8169   0.000412
#   U1 2022 (bear)      5m        20       1617   0.000891
#   U1 2022 (bear)     15m        20        525   0.001478
#   U1 2026 (recent)    1m        19       7780   0.000282
#   U1 2026 (recent)    5m        20       1540   0.000585
#   U1 2026 (recent)   15m        20        500   0.001010
#   crypto 2023+        5m        20     385902   0.001221
#   crypto 2023+       15m        20     128633   0.002034
#
# **The answer is that the label has material to work with.** The kill criterion was an order of
# magnitude below crypto; the gap is a factor of 1.4x (2022) to 2.1x (2026) at both 5m and 15m —
# the same order. Twenty US ETFs disperse less than twenty altcoins, and not much less.
#
# Two things to carry forward rather than forget. The recent regime is the thinnest of the three
# (0.000585 at 5m against 0.000891 in 2022), so a model fitted and judged only on recent years
# faces the hardest version of this problem — which is an argument for the four folds, not against
# them. And dispersion grows roughly as the square root of the bar (1m 0.000412 -> 5m 0.000891 ->
# 15m 0.001478, against sqrt ratios of 2.24 and 1.73): moves accumulate close to independently
# across the basket, which is what a dominant single factor would not do.
SD_T_CRYPTO_5M = 0.001221  # the scale every equity figure above is quoted against


def within_session(close: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Log returns over `rule`, resampled per session and never across one.

    Grouped by exchange-local date before resampling, so a bucket never spans two sessions and the
    first return of each day — the overnight gap — is dropped rather than counted as a bar.
    """
    day = close.index.tz_convert(EXCHANGE).date
    out = []
    for _, session in close.groupby(day):
        sampled = session.resample(rule).last().dropna(how="all")
        out.append(np.log(sampled).diff().iloc[1:])  # iloc[1:] is the gap, and it is not a bar
    return pd.concat(out) if out else pd.DataFrame()


def sd_t(returns: pd.DataFrame, minimum: int = 2) -> pd.Series:
    """Cross-sectional standard deviation per timestamp — the label's denominator.

    `minimum` guards the tail of a sample: a timestamp where only one symbol traded has a standard
    deviation of NaN, and one where two did has a very noisy one that would still enter a median.
    """
    return returns.std(axis=1).where(returns.notna().sum(axis=1) >= minimum).dropna()


def equity_panel(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """1m closes of `symbols` over one window, session-filtered, one column per symbol.

    Fetched at 1m and resampled upwards by `within_session`, which is the store's own rule (§3):
    the minute bar is native, 5m and 15m derive from it, and the reverse is not possible.
    """
    out = {}
    for symbol in symbols:
        frame = bars(symbol, "1m", start, end)
        if not frame.empty:
            out[symbol] = frame.close
    return pd.DataFrame(out).sort_index()


def crypto_panel(store: Path = CRYPTO_STORE, start: str | None = None, end: str | None = None):
    """5m closes of the previous project's twenty pairs, read straight from its Parquet store."""
    out = {}
    for path in sorted(store.glob("*USDT-5m.parquet")):
        close = pd.read_parquet(path, columns=["close"]).close
        out[path.stem.split("USDT")[0]] = close.loc[start:end]
    return pd.DataFrame(out).sort_index()


def summarise(close: pd.DataFrame, rule: str, label: str, session: bool = True) -> dict:
    """One row: the median `sd_t` and what it was computed over."""
    if session:
        returns = within_session(close, rule)
    else:
        # Crypto never closes, so there is no session to stay inside and no gap to drop.
        returns = np.log(close.resample(rule).last()).diff().iloc[1:]
    spread = sd_t(returns)
    return {
        "set": label,
        "bar": rule.replace("min", "m"),
        "symbols": int(returns.notna().sum(axis=1).median()),
        "instants": len(spread),
        "sd_t": round(float(spread.median()), 6),
        "sd_t_p25": round(float(spread.quantile(0.25)), 6),
        "sd_t_p75": round(float(spread.quantile(0.75)), 6),
    }


def _selfcheck() -> None:
    """The session boundary and the dispersion, on a panel built here — fetching needs the network."""
    # Two sessions of 1m bars, with a violent overnight gap between them. Both symbols gap the same
    # way, so the gap carries no cross-sectional dispersion of its own — but it does carry a return
    # ten times a bar's, and a resampling that spans the close would smear it into a bucket.
    grid = pd.DatetimeIndex(
        list(pd.date_range("2026-03-10 09:30", periods=390, freq="1min", tz=EXCHANGE))
        + list(pd.date_range("2026-03-11 09:30", periods=390, freq="1min", tz=EXCHANGE))
    ).tz_convert("UTC")
    rng = np.random.default_rng(0)
    steps = rng.normal(scale=1e-4, size=(780, 2))
    steps[390] = [0.10, 0.10]  # the gap: same for both, ten times a bar
    close = pd.DataFrame(np.exp(np.cumsum(steps, axis=0)), index=grid, columns=["A", "B"])

    returns = within_session(close, "5min")
    assert returns.abs().max().max() < 0.01, "the overnight gap is not a bar and must not appear"
    # 390 one-minute bars make 78 five-minute buckets; the first of each session is the gap.
    assert len(returns) == 2 * (78 - 1), f"two sessions of 77 usable buckets, got {len(returns)}"
    days = pd.Series(returns.index.tz_convert(EXCHANGE).date).nunique()
    assert days == 2, "a bucket never spans two sessions"

    # Dispersion rises with the bar, because a longer bar accumulates more independent moves.
    assert summarise(close, "15min", "x")["sd_t"] > summarise(close, "1min", "x")["sd_t"]

    # `sd_t` is a cross-section, so a timestamp carrying one symbol has none and is dropped.
    one = returns.copy()
    one.iloc[0, 1] = np.nan
    assert len(sd_t(one)) == len(returns) - 1
    # Two identical columns have zero dispersion — the degenerate cross-section, priced.
    same = pd.DataFrame({"A": returns.A, "B": returns.A})
    assert float(sd_t(same).max()) == 0.0, "clones carry no dispersion, which is the whole worry"

    # A crypto-style panel is contiguous: there is no close to stay inside, so nothing is dropped
    # and a large move is kept. Built separately because the equity fixture has a hole in it, and a
    # hole makes the two paths agree for the wrong reason — the gap return is NaN either way.
    always = pd.date_range("2026-03-10", periods=576, freq="5min", tz="UTC")
    steps = rng.normal(scale=1e-4, size=(576, 2))
    steps[300] = [0.10, -0.10]  # a move of ten percent, in opposite directions
    continuous = pd.DataFrame(np.exp(np.cumsum(steps, axis=0)), index=always, columns=["A", "B"])
    flat = np.log(continuous.resample("5min").last()).diff().iloc[1:]
    assert len(flat) == 575, "a contiguous panel drops nothing but the first diff"
    assert abs(sd_t(flat).max() - 0.1 * np.sqrt(2)) < 1e-6, "and the big cross-sectional move is kept"
    # Which is the whole difference: the session path would have cut a move at a boundary.
    assert summarise(continuous, "5min", "x", session=False)["instants"] == 575


if __name__ == "__main__":
    import sys

    if "--run" in sys.argv:
        from alpacatrader.universe import U1

        rows = []
        for name, (start, end) in PERIODS.items():
            panel = equity_panel(list(U1.values()), start, end)
            for rule in RULES:
                rows.append(summarise(panel, rule, f"U1 {name}"))
        crypto = crypto_panel(start="2023-01-01", end="2026-09-01")
        for rule in RULES[1:]:
            rows.append(summarise(crypto, rule, "crypto 2023+", session=False))
        out = pd.DataFrame(rows)
        print(out.to_string(index=False))
    else:
        _selfcheck()
        print("ok — no return crosses a session, and clones carry no dispersion")
