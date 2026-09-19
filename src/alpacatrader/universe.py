"""Measure 1a: the twenty exposures, chosen on how differently they move.

The label is an ordering **inside the instant**, divided by the dispersion of that instant. Twenty
clones of the S&P have a degenerate cross-section and the label becomes noise divided by almost
nothing, so what the universe has to supply is twenty structures that move apart. Cost does not
enter here: it decides *which instrument* buys an exposure already chosen (measure 1b) and it makes
the simulations honest (`costs`), but a dataset chosen by what is cheap to trade is a dataset chosen
by the wrong criterion.

This is also the priority §4 of the spec leaves open. It says criteria 1 (high price per share) and
5 (real cross-sectional dispersion) are in conflict and that the lever is the existence of two
issuers for the same exposure. The order is: **criterion 5 picks the exposures, criterion 1 picks
the issuer inside one.**

**Estimated on the first fold and verified, never refitted, on the rest.** Choosing exposures by
maximising decorrelation over the whole history is a decision taken on the test slice, which is the
one thing the project forbids everywhere else (§8). A universe that knows which sectors decoupled in
2022 knows something about 2022. `ESTIMATION_END` is the cut, and `report` prints the same numbers
on the later periods so a set that only decorrelated in-sample is visible as such.

    uv run python -m alpacatrader.universe
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpacatrader.data.candles import get_candles

# One entry per *exposure*, each listing the instruments that replicate it — the SPDR first, the
# Vanguard (or nearest) alternative second where one exists. Measure 1b chooses between them on
# measured spread; here any member stands for the exposure, because two funds tracking one index
# correlate at ~0.99 and which one represents it cannot change a correlation matrix.
#
# Deliberately wider than eleven sectors: bonds, metals, credit, the dollar and international equity
# are where the decorrelation actually lives, and a basket of US sector ETFs alone is the degenerate
# cross-section this measure exists to avoid.
EXPOSURES: dict[str, list[str]] = {
    "large_cap": ["SPY", "VOO"],
    "nasdaq": ["QQQ"],
    "small_cap": ["IWM", "VB"],
    "utilities": ["XLU", "VPU"],
    "materials": ["XLB", "VAW"],
    "real_estate": ["XLRE", "VNQ"],
    "energy": ["XLE", "VDE"],
    "financials": ["XLF", "VFH"],
    "staples": ["XLP", "VDC"],
    "discretionary": ["XLY", "VCR"],
    "industrials": ["XLI", "VIS"],
    "healthcare": ["XLV", "VHT"],
    "communications": ["XLC", "VOX"],
    "technology": ["XLK", "VGT"],
    "semiconductors": ["SMH", "SOXX"],
    "biotech": ["XBI", "IBB"],
    "regional_banks": ["KRE"],
    "oil_gas_e_p": ["XOP"],
    "homebuilders": ["ITB", "XHB"],
    "retail": ["XRT"],
    "gold": ["GLD", "IAU"],
    "silver": ["SLV"],
    "long_treasury": ["TLT", "VGLT"],
    "short_treasury": ["SHY", "VGSH"],
    "inflation_linked": ["TIP", "VTIP"],
    "high_yield": ["HYG", "JNK"],
    "emerging": ["EEM", "VWO"],
    "developed_ex_us": ["EFA", "VEA"],
    "china": ["FXI"],
    "dollar": ["UUP"],
}

# The floor of Alpaca's history, and the first fold's cut. Everything that decides is measured
# before it; everything after it is a check that the decision survived out of sample.
HISTORY_START = "2016-01-04"
ESTIMATION_END = "2019-12-31"

# Which exposures are a *slice* of another, as a matter of what the funds hold and not of what the
# correlations came out at. KRE is inside XLF, XBI inside XLV, SMH inside XLK, XOP inside XLE, ITB
# and XRT inside XLY. A pair like this is not two exposures, so both may not sit in E1 at once.
#
# The rule only bites when **both** land in E1 — a slice whose container was not selected is a
# perfectly good exposure of its own, which is why SMH stays eligible while XLK is out. When both
# do land, the container wins: it is the more liquid and the more capacious of the two, the §4
# criteria on volume and history point that way, and the slice's extra dispersion is not worth a
# column that carries a duplicate rank.
CONTAINED_IN = {
    "regional_banks": "financials",
    "biotech": "healthcare",
    "semiconductors": "technology",
    "oil_gas_e_p": "energy",
    "homebuilders": "discretionary",
    "retail": "discretionary",
}

SIZE = 20  # the cardinality is fixed: changing it retunes every significance threshold in §8

# Measure 1a, taken 2026-09-19 on daily SIP closes from 2016-01-04, estimated on the first fold
# (to 2019-12-31) and only checked after it. Reproduce with `python -m alpacatrader.universe --run`.
#
#   period       set             n   mean_corr   max_corr    sd_t
#   estimation   E1             20      0.3537     0.7752   0.00748
#   estimation   sectors_only   12      0.5163     0.8600   0.00593
#   after        E1             20      0.3517     0.8407   0.01033
#   after        sectors_only   12      0.5398     0.8987   0.00846
#
# The number that decides: 0.3537 in the estimation window and 0.3517 after it. The decorrelation
# is a property of these exposures and not a fit to the window they were chosen on — which is the
# failure this measure was built to be able to see. Against the twelve-sector basket §4 warns about,
# the cross-sectional spread the label divides by is 26% wider in-sample and 22% wider after.
#
# The worst surviving pair is XLB/XLU at 0.841 after the cut, from the rate cycle rather than from
# overlapping holdings. Named here because it is E1's weakest joint and the first thing to revisit
# if M3.5 finds the intraday dispersion thin.
E1 = [
    "semiconductors",
    "dollar",
    "utilities",
    "short_treasury",
    "oil_gas_e_p",
    "silver",
    "financials",
    "homebuilders",
    "retail",
    "staples",
    "healthcare",
    "high_yield",
    "real_estate",
    "china",
    "materials",
    "long_treasury",
    "communications",
    "developed_ex_us",
    "small_cap",
    "inflation_linked",
]


def representative(exposure: str) -> str:
    """The instrument standing for an exposure while it is being chosen. Measure 1b replaces it."""
    return EXPOSURES[exposure][0]


def closes(symbols: list[str], days: int) -> pd.DataFrame:
    """Daily closes, one column per symbol, on the union calendar.

    Daily and not intraday on purpose: this measure is about whether two exposures *are* different
    things, which a decade of daily bars answers at the cost of one request each. Whether they stay
    different at the frequency the book trades is a separate question and a separate measure, on the
    intraday store (M3.5) — intraday the common market factor dominates far more than it does here,
    so a set that passes this can still fail that.
    """
    out = {}
    for symbol in symbols:
        bars = get_candles(symbol, "1d", days)
        if not bars.empty:
            out[symbol] = bars.close
    return pd.DataFrame(out).sort_index()


def correlations(close: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation of daily log returns.

    On returns and never on prices: two rising series correlate near 1 whatever their returns do,
    and it is the returns a cross-sectional ranking is taken over.
    """
    return np.log(close).diff().corr()


def _greedy(c: pd.DataFrame, size: int, banned: set[str]) -> list[str]:
    """Minimax admission over the columns of `c` that are not banned."""
    pool = c.columns.difference(list(banned))
    c = c.loc[pool, pool]
    if len(pool) < 2:
        return list(pool)
    chosen = list(c.stack().idxmin())  # the two that agree least, whatever else happens
    while len(chosen) < size:
        rest = c.columns.difference(chosen)
        if rest.empty:
            break
        chosen.append(c.loc[rest, chosen].max(axis=1).idxmin())
    return chosen


def select(corr: pd.DataFrame, size: int = SIZE, contained=CONTAINED_IN, names=None) -> list[str]:
    """`size` columns chosen greedily to keep the worst pairwise correlation low.

    Minimax rather than mean: one pair at 0.99 ruins a cross-section that averages well, because the
    two carry the same rank and the label cannot tell them apart. Seeded with the least correlated
    pair, then each step adds whichever candidate has the smallest maximum correlation to what is
    already chosen.

    Greedy and not exhaustive: choosing 20 of 30 is 30 million subsets, and the ordering this
    produces is stable enough that the last admitted members are visibly the marginal ones — which
    is the information a human needs to overrule it.

    Then the containment repair, which correlation cannot see: if a set comes back holding both a
    container and its slice, the slice is banned and the whole admission is taken again — not
    patched by swapping the tail, because banning a member that was admitted early changes every
    choice after it. It runs to a fixed point and terminates, since each pass bans at least one.
    """
    c = corr.abs().copy()
    np.fill_diagonal(c.values, np.nan)
    names = {col: col for col in c.columns} if names is None else names
    banned: set[str] = set()
    while True:
        chosen = _greedy(c, size, banned)
        inside = {names[col] for col in chosen}
        extra = {col for col in chosen if contained.get(names[col]) in inside}
        if not extra:
            return chosen
        banned |= extra


def dispersion(close: pd.DataFrame, columns: list[str]) -> float:
    """Median cross-sectional standard deviation of daily returns — the label's denominator.

    The quantity the whole exercise is for: a ranking inside an instant divided by the spread of
    that instant. Reported next to the correlations because a set can look decorrelated pairwise and
    still move together on the days that matter.
    """
    return float(np.log(close[columns]).diff().std(axis=1).median())


def report(close: pd.DataFrame, chosen: list[str]) -> pd.DataFrame:
    """The chosen set against the full candidate list, in-sample and after the cut.

    Two rows that matter: `chosen` after `ESTIMATION_END` is the out-of-sample check that the
    decorrelation was a property and not a fit, and `sectors_only` is the degenerate cross-section
    the selection exists to avoid, priced so the gain has a number.
    """
    sectors = [representative(e) for e in list(EXPOSURES)[3:15]]  # the eleven GICS sectors + nasdaq
    periods = {"estimation": close.loc[:ESTIMATION_END], "after": close.loc[ESTIMATION_END:]}
    sets = {"chosen": chosen, "all_candidates": list(close.columns), "sectors_only": sectors}
    rows = []
    for period, frame in periods.items():
        for name, cols in sets.items():
            cols = [c for c in cols if c in frame.columns]
            c = correlations(frame[cols]).abs()
            np.fill_diagonal(c.values, np.nan)
            rows.append(
                {
                    "period": period,
                    "set": name,
                    "n": len(cols),
                    "mean_corr": round(float(c.stack().mean()), 4),
                    "max_corr": round(float(c.stack().max()), 4),
                    "sd_t": round(dispersion(frame, cols), 5),
                }
            )
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    """Take the measure again and print it. The decision is `E1`; this is how it was reached."""
    names = {representative(e): e for e in EXPOSURES}
    close = closes(list(names), days=4000)
    chosen = select(correlations(close.loc[:ESTIMATION_END]), names=names)
    print("E1:", ", ".join(f"{names[c]}({c})" for c in chosen))
    out = report(close, chosen)
    print(out.to_string(index=False))
    if [names[c] for c in chosen] != E1:
        print("\nWARNING: the selection no longer reproduces E1 — a later inception or a revised")
        print("candidate list has moved it. Do not edit E1 silently; the constant is a decision.")
    return out


def _selfcheck() -> None:
    """The selection, on a correlation matrix with a known answer — the fetch needs the network."""
    _consistency()
    rng = np.random.default_rng(0)
    # Three blocks of four near-identical series plus two independent ones: any honest minimax pick
    # of five takes at most one member per block, because a second one costs 0.99.
    factors = rng.normal(size=(2000, 5))
    cols, data = [], []
    for block in range(3):
        for member in range(4):
            cols.append(f"b{block}m{member}")
            data.append(factors[:, block] + 0.05 * rng.normal(size=2000))
    for extra in range(2):
        cols.append(f"x{extra}")
        data.append(factors[:, 3 + extra])
    close = pd.DataFrame(np.exp(np.cumsum(np.array(data).T, axis=0)), columns=cols)

    corr = correlations(close)
    chosen = select(corr, size=5)
    assert len(chosen) == 5 and len(set(chosen)) == 5
    blocks = [c[:2] for c in chosen if c.startswith("b")]
    assert len(blocks) == len(set(blocks)), f"two members of one block chosen: {chosen}"
    assert {"x0", "x1"} <= set(chosen), f"the independent series must be in: {chosen}"
    worst = correlations(close[chosen]).abs().to_numpy()
    np.fill_diagonal(worst, 0.0)
    assert worst.max() < 0.3, f"the chosen set still has a pair at {worst.max():.2f}"

    # A whole-block selection is what the greedy step exists to beat, and it has to be visibly worse.
    assert dispersion(close, chosen) > dispersion(close, cols[:5]), "spread out beats one block"
    # Asking for more than there is returns what there is rather than looping for ever.
    assert len(select(corr, size=99)) == len(cols)

    # --- the containment repair -----------------------------------------------------------------
    # `x0` is declared a slice of `x1`, and the two are independent, so correlation alone would
    # never separate them: only the declaration can. The container stays, the slice goes, and the
    # freed seat goes to a block member rather than being left empty.
    both = select(corr, size=5, contained={"x0": "x1"})
    assert "x1" in both and "x0" not in both, f"the container wins, the slice goes: {both}"
    assert len(both) == 5, "and the freed seat is refilled"
    # A slice whose container was not selected keeps its place: the rule bites on pairs in the set,
    # never on a name. Declaring `x0` inside something the greedy did not take changes nothing.
    base = select(corr, size=5)
    outside = next(col for col in cols if col not in base)
    assert select(corr, size=5, contained={"x0": outside}) == base, "an absent container does not bite"


def _consistency() -> None:
    """`E1` must name exposures that exist and carry no container/slice pair. No network."""
    assert len(E1) == SIZE and len(set(E1)) == SIZE
    assert not set(E1) - set(EXPOSURES), f"E1 names unknown exposures: {set(E1) - set(EXPOSURES)}"
    inside = {e for e in E1 if CONTAINED_IN.get(e) in set(E1)}
    assert not inside, f"E1 holds both a container and its slice: {inside}"
    assert all(EXPOSURES[e] for e in E1), "every exposure needs at least one instrument"


if __name__ == "__main__":
    import sys

    if "--run" in sys.argv:
        run()
    else:
        _selfcheck()
        _consistency()
        print("ok — the selection keeps one per block, and E1 is self-consistent")
