"""Measure 2: what one side of a trade costs, per symbol.

With commission at zero the cost per side is dominated by the half spread, which in basis points is
inversely proportional to the price per share. Between SPY at $761 and XLU at $41 there are **eight
times the cost** at the same quoted penny. A scalar `FEE` does not describe this basket, and that is
the whole reason this module exists.

    cost_per_side(bp) = 0.103 + 0.975/P + 5000·s/P
                        SEC/2    TAF/2    half spread

The first term is proportional to value and identical for every symbol; the other two are per share
and scale as 1/P.

**The rates are configuration, never constants.** SEC §31 was $0.00/M until 2026-04-03 and changes
every year; the TAF has its own schedule and its own cap. A rate hard-coded in a formula is a number
nobody re-reads when the notice changes, so they live in `Rates` and travel as an argument.

**Where this belongs, and where it does not.** Cost makes the simulations honest and it chooses the
instrument for an exposure (`universe`, measure 1b). It does **not** choose the exposures: a dataset
picked by what is cheap to trade is a dataset picked by the wrong criterion.

**How it reaches the model.** `swing.fit_policy` already pays the fee inside the reward,
`reward = p·r − fee·|p − before|`. `fee` becoming a vector per symbol is enough for the policy to
learn on its own not to churn XLU and to churn SPY. A change of type, not of architecture — and the
same for `--fee`, already a CLI argument in `swing`, `threshold`, `stops`, `swingrule` and `oracle`:
it changes who supplies it, not who consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

BP = 1e4  # basis points in a unit fraction


@dataclass(frozen=True)
class Rates:
    """The regulatory charges, both on the sell side only.

    Defaults are the Alpaca fee schedule of 2026-09-01 and the FINRA notices behind it. Both are
    dated because both expire: SEC §31 is reset annually and read $0.00/M as recently as April 2026.
    """

    sec_per_dollar: float = 20.60 / 1e6  # SEC §31, sell only — FINRA Information Notice 03/17/26
    taf_per_share: float = 0.000195  # FINRA Trading Activity Fee, sell only, from 2026-01-01
    taf_cap: float = 9.79  # per trade, which is 50,205 shares
    commission: float = 0.0  # Alpaca: $0 on every US-listed stock and ETF, no exceptions


RATES = Rates()


def taf(shares: float, rates: Rates = RATES) -> float:
    """The TAF actually charged on a sale of `shares`, cap included.

    Per *share* and not per dollar, which is why it hits cheap stocks and not small tickets, and
    why it needs the share count rather than the notional.
    """
    return min(shares * rates.taf_per_share, rates.taf_cap)


def per_side_bp(price: float, spread: float, rates: Rates = RATES) -> float:
    """Cost of one side in basis points, at `price` dollars per share and `spread` dollars quoted.

    Averaged over the two sides: SEC and TAF fall on the sell only, so each contributes half of its
    round-trip rate to a figure quoted per side. The half spread is paid on both sides and enters
    whole. The cap is ignored here — it binds above 50,205 shares, which at these prices is a ticket
    far larger than anything this book sends, and `taf` is where a real ticket is priced.
    """
    sec = rates.sec_per_dollar * BP / 2
    per_share = rates.taf_per_share * BP / 2 + spread * BP / 2
    return rates.commission * BP + sec + per_share / price


def table(prices: pd.Series, spreads: pd.Series, rates: Rates = RATES) -> pd.DataFrame:
    """Cost per side for a whole basket, sorted cheapest first.

    `spreads` in dollars, aligned to `prices` by symbol. The `x_spy` column is the one to read: it
    says how many times more a rotation costs here than on the cheapest instrument in the book, and
    it is the number that decides an issuer in measure 1b.
    """
    bp = pd.Series({s: per_side_bp(prices[s], spreads[s], rates) for s in prices.index})
    out = pd.DataFrame({"price": prices, "spread": spreads, "bp": bp.round(3)}).sort_values("bp")
    return out.assign(x_cheapest=(out.bp / out.bp.min()).round(1))


def short_dividend_bp(annual_yield: float, days: int) -> float:
    """What a short position pays in dividends over `days`, in basis points.

    A cost and not a price correction. `adjustment=all` makes the series total-return, so the
    dividend is already inside the price a long earns — but a short *pays* it at the ex-date, and
    the adjusted series shows nothing. On a market-neutral book with a permanent short leg this is
    1.5–3% a year the backtest would otherwise never charge.
    """
    return annual_yield * BP * days / 365


def flat(symbols, bp: float) -> pd.Series:
    """One rate for every symbol, as a vector.

    The bridge to every number the previous project measured: those runs charged a scalar, so a
    scalar has to survive as the degenerate case of the vector. If this and a per-symbol vector of
    equal entries ever disagree, the change of type broke the arithmetic rather than generalising
    it — which is what the self-check asserts.
    """
    return pd.Series(float(bp), index=list(symbols))


def _selfcheck() -> None:
    """The published table, and the flat case that must not have moved."""
    # §2 of the spec, at a quoted penny: real prices of 2026-09-19 and the cost they imply. These
    # are the numbers the venue decision was taken on, so the formula has to still produce them.
    published = {
        "SPY": (761.69, 0.170),
        "QQQ": (721.45, 0.174),
        "XLK": (189.60, 0.372),
        "XLV": (168.39, 0.406),
        "XLE": (64.31, 0.896),
        "XLF": (55.86, 1.016),
        "XLB": (49.99, 1.123),
        "XLRE": (42.53, 1.302),
        "XLU": (41.10, 1.344),
    }
    # Tolerance 1e-3 and not tighter, for a reason worth writing down rather than tuning away: the
    # spec prints each component to three decimals and totals the *rounded* parts, so XLU reads
    # 0.103 + 0.024 + 1.217 = 1.344 where the exact sum is 1.3433. The gap is the table's rounding,
    # and it is largest exactly where the 1/P terms are largest.
    for symbol, (price, expected) in published.items():
        got = per_side_bp(price, 0.01)
        assert abs(got - expected) < 1e-3, f"{symbol}: {got:.4f} bp against the spec's {expected}"

    # The spread is what the basket's spread of cost is made of: eight times between SPY and XLU.
    assert round(per_side_bp(41.10, 0.01) / per_side_bp(761.69, 0.01), 1) == 7.9

    # The three terms separate cleanly, which is what makes the basket's spread of cost readable:
    # the SEC term does not move with the price at all, and the half spread is exactly 5000·s/P.
    assert abs(per_side_bp(100.0, 0.0) - per_side_bp(500.0, 0.0) - 0.975 * (1 / 100 - 1 / 500)) < 1e-12
    assert abs(per_side_bp(100.0, 0.02) - per_side_bp(100.0, 0.0) - 5000 * 0.02 / 100) < 1e-12
    # Only the *spread* part is invariant under scaling both together — the TAF term still falls as
    # 1/P — so a penny on a $41 share is not four cents on a $164 one. They differ by 0.018 bp.
    assert abs(per_side_bp(41.0, 0.01) - per_side_bp(164.0, 0.04) - 0.975 * (1 / 41 - 1 / 164)) < 1e-12

    # A rate is configuration: the SEC term read zero until April 2026 and the formula has to follow
    # it there rather than carry a baked-in 0.103.
    old = Rates(sec_per_dollar=0.0)
    assert abs(per_side_bp(761.69, 0.01, old) - (0.170 - 0.103)) < 5e-4

    # The cap is per trade and binds at 50,205 shares — below it the TAF is linear, above it flat.
    assert abs(taf(1_000) - 0.195) < 1e-12
    assert taf(50_000) < RATES.taf_cap == taf(60_000) == taf(10_000_000)

    # The flat vector is the old scalar: a per-symbol fee of equal entries has to be the same number
    # everywhere, or the change of type moved the arithmetic instead of generalising it.
    symbols = ["SPY", "XLU", "TLT"]
    assert flat(symbols, 25.0).equals(pd.Series(25.0, index=symbols))
    assert flat(symbols, 25.0).nunique() == 1 and float(flat(symbols, 25.0).iloc[0]) == 25.0

    # The dividend on a short is a cost that grows with the holding period and nothing else.
    assert abs(short_dividend_bp(0.02, 365) - 200.0) < 1e-9
    assert abs(short_dividend_bp(0.02, 1) - 200.0 / 365) < 1e-9
    assert short_dividend_bp(0.02, 0) == 0.0

    prices = pd.Series({"SPY": 761.69, "XLU": 41.10})
    out = table(prices, pd.Series({"SPY": 0.01, "XLU": 0.01}))
    assert list(out.index) == ["SPY", "XLU"], "cheapest first"
    assert out.x_cheapest.iloc[-1] == 7.9


if __name__ == "__main__":
    _selfcheck()
    print("ok — the published table reproduces, and a flat vector is the old scalar")
