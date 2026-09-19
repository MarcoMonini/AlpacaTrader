"""Measure 5, second half: is the gross the rule's timing, or the time it spends in the market?

The question `buy_and_hold` cannot answer. It holds every bar, so the comparison mixes two things:
a rule in the market half the time earns roughly half the drift *whatever it picks*. On the
previous venue the basket fell 49% and a short-biased rule looked skilled; **here the basket rises
and a long-biased one will look skilled**, which is the same mistake with the sign flipped. §9 lists
it among the things not to do again, so the control is in from the first line rather than added at
the end.

`rotation_null` rotates each symbol's position vector by a random offset. The same bars are held,
the same number of trades, the same holding-period structure — only the alignment with price is
destroyed. What is left in the spread is the null the real gross has to clear. On crypto it read
real −0.4151 against a null of −0.4098 ± 0.0949, z −0.06: the entry timing was indistinguishable
from rolling the same positions to a random phase, and the whole of the gross was exposure.

**Rotation, never a reshuffle.** Shuffling bar by bar breaks the runs into noise and compares
against a null that trades thousands of times a year instead of dozens. Only a rotation keeps the
trade structure and moves the phase.

**Rotated inside the tradable rows, which is this project's adaptation.** The previous project
rolled the whole vector because its market never closed. Here a roll over all rows would drop
positions onto bars the book must be flat on — the warm-up at the open and the unclosed leg at the
close — so the null would hold exposure the rule cannot. The roll is taken over the usable rows
only, which preserves every run and keeps every position on a bar that could have been traded.

**Flat at every close, and it is paid for.** Intraday-only means the book enters from zero each
morning and exits to zero each afternoon, so both legs are charged at the per-symbol cost of §2 —
not a scalar. A rule holding four names a side pays eight crossings a session before it has
predicted anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpacatrader import columns, panel
from alpacatrader.costs import per_side_bp

THETA = 0.2  # top and bottom fifth: four symbols a side out of twenty, the widest sensible book
DRAWS = 500  # §9 asks for at least 500 rotations before quoting a z
SESSIONS_PER_YEAR = 252


def positions(signal: pd.DataFrame, usable: pd.Series, theta: float = THETA, sign: int = 1) -> pd.DataFrame:
    """A dollar-neutral book: long the top `theta` of the cross-section, short the bottom, flat else.

    `sign` flips the reading, because a column whose Rank IC is negative is traded the other way
    round — `stretch` and `streak` both lead the price downwards, so the tradable rule buys the
    *least* stretched. Weights are 1/n a side so the two legs are the same size whatever the
    cross-section's width that instant.
    """
    masked = signal.where(np.broadcast_to(usable.to_numpy()[:, None], signal.shape))
    ranked = (sign * masked).rank(axis=1, pct=True)
    long, short = ranked > 1 - theta, ranked <= theta
    weights = long.astype(float).div(long.sum(axis=1).replace(0, np.nan), axis=0) - short.astype(float).div(
        short.sum(axis=1).replace(0, np.nan), axis=0
    )
    return weights.fillna(0.0)


def turnover(pos: pd.DataFrame, session: pd.Series) -> pd.DataFrame:
    """Absolute weight traded per bar, entering from flat each open and exiting to flat each close."""
    same = (session == session.shift()).to_numpy()
    previous = pos.shift().where(np.broadcast_to(same[:, None], pos.shape), 0.0)
    traded = (pos - previous).abs()
    last = (session != session.shift(-1)).to_numpy()
    return traded + pos.abs().where(np.broadcast_to(last[:, None], pos.shape), 0.0)


def gross(pos: pd.DataFrame, ret: pd.DataFrame, years: float) -> float:
    """Log return per year of the book, before cost."""
    return float((pos * ret).sum(axis=1).sum() / years)


def cost(pos: pd.DataFrame, session: pd.Series, fee: pd.Series, years: float) -> float:
    """What the book pays per year, at the per-symbol basis points of §2."""
    return float((turnover(pos, session) * fee.reindex(pos.columns) / 1e4).sum(axis=1).sum() / years)


def rotation_null(pos: pd.DataFrame, ret: pd.DataFrame, usable: pd.Series, years: float, draws=DRAWS, seed=0):
    """The gross of the same positions rolled to a random phase, `draws` times.

    Each symbol gets its own offset, so the cross-sectional structure is broken as well as the
    temporal one — a null that rolled every symbol together would still be a coherent book, just a
    late one.
    """
    keep = usable.to_numpy()
    held, future = pos.to_numpy()[keep], np.nan_to_num(ret.to_numpy()[keep])
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for draw in range(draws):
        offsets = rng.integers(1, len(held), size=held.shape[1])
        rolled = np.stack([np.roll(held[:, j], offsets[j]) for j in range(held.shape[1])], axis=1)
        out[draw] = (rolled * future).sum() / years
    return out


def price(name: str, signal: pd.DataFrame, p: dict, fee: pd.Series, sign: int = 1, draws=DRAWS) -> dict:
    """One row of the answer: gross, null, z, cost and net for one column's book."""
    usable, session = p["usable"], p["session"]
    years = session.nunique() / SESSIONS_PER_YEAR
    ret = panel.forward(p, 1)
    pos = positions(signal, usable, sign=sign)
    real = gross(pos, ret, years)
    null = rotation_null(pos, ret, usable, years, draws=draws)
    spent = cost(pos, session, fee, years)
    trades = float(turnover(pos, session).sum().sum() / years)
    return {
        "column": name,
        "sign": sign,
        "gross": round(real, 4),
        "null": round(float(null.mean()), 4),
        "null_sd": round(float(null.std()), 4),
        "z": round((real - float(null.mean())) / float(null.std()), 2) if null.std() > 0 else np.nan,
        "cost": round(spent, 4),
        "net": round(real - spent, 4),
        "turnover": round(trades, 1),
    }


def fees(p: dict) -> pd.Series:
    """Cost per side in basis points per symbol, from the measured spreads of M1b."""
    from alpacatrader.universe import U1

    measured = {
        "SMH": (560.76, 0.07),
        "UUP": (28.38, 0.01),
        "XLU": (41.47, 0.01),
        "SHY": (81.32, 0.01),
        "XOP": (192.40, 0.07),
        "SLV": (59.36, 0.01),
        "VFH": (136.39, 0.02),
        "XHB": (97.25, 0.04),
        "XRT": (83.06, 0.01),
        "XLP": (83.40, 0.01),
        "XLV": (168.40, 0.02),
        "JNK": (94.58, 0.01),
        "VNQ": (93.88, 0.01),
        "FXI": (34.20, 0.01),
        "XLB": (50.74, 0.01),
        "TLT": (81.22, 0.01),
        "XLC": (112.44, 0.01),
        "EFA": (105.76, 0.01),
        "IWM": (286.08, 0.01),
        "TIP": (105.58, 0.01),
    }
    assert set(measured) == set(U1.values()), "the fee vector must cover U1 exactly"
    return pd.Series({s: per_side_bp(*measured[s]) for s in measured})


def _selfcheck() -> None:
    """A book that knows the future, one that does not, and the null between them."""
    per, days, k = 40, 40, 6
    n = per * days
    rng = np.random.default_rng(0)
    session = pd.Series(np.repeat(np.arange(days), per))
    bar = pd.Series(np.tile(np.arange(per), days))
    close = pd.DataFrame(np.exp(np.cumsum(rng.normal(scale=1e-3, size=(n, k)), axis=0)), columns=list("abcdef"))
    p = {
        "close": close,
        "session": session,
        "bar": bar,
        "size": session.map(session.value_counts()),
        "usable": pd.Series(np.ones(n, dtype=bool)),
    }
    ret = panel.forward(p, 1)
    years = days / SESSIONS_PER_YEAR

    # A dollar-neutral book: every row sums to zero and each leg is one unit.
    pos = positions(pd.DataFrame(rng.normal(size=(n, k)), columns=close.columns), p["usable"])
    assert np.allclose(pos.sum(axis=1), 0.0, atol=1e-12)
    assert np.allclose(pos[pos > 0].sum(axis=1).dropna(), 1.0)

    # A book that can see the next bar makes money; the same book rolled to a random phase does not.
    oracle = positions(ret.shift(-0), p["usable"])  # ret at row i is the return over (i, i+1]
    assert gross(oracle, ret, years) > 0
    null = rotation_null(oracle, ret, p["usable"], years, draws=50)
    z = (gross(oracle, ret, years) - null.mean()) / null.std()
    assert z > 5, f"a book that knows the future must beat its own rotation, z was {z:.1f}"
    assert abs(null.mean()) < abs(gross(oracle, ret, years)) / 5, "the null holds no timing"

    # A book built on noise does not beat its own rotation — the case the measure exists for.
    blind = positions(pd.DataFrame(rng.normal(size=(n, k)), columns=close.columns), p["usable"])
    noise_null = rotation_null(blind, ret, p["usable"], years, draws=200)
    assert abs((gross(blind, ret, years) - noise_null.mean()) / noise_null.std()) < 3

    # Turnover: flat at every open and every close, so a session that holds one book pays twice.
    steady = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    steady.iloc[:, 0] = 0.5
    turned = turnover(steady, session)
    assert np.isclose(turned.iloc[0, 0], 0.5), "entering from flat"
    assert np.isclose(turned.iloc[per - 1, 0], 0.5), "and exiting to flat at the close"
    assert np.isclose(turned.iloc[1, 0], 0.0), "holding costs nothing in between"
    assert np.isclose(turned.sum().sum(), 0.5 * 2 * days), "two crossings a session, every session"

    # Cost is per symbol, so the same book on a dearer name costs more.
    cheap = pd.Series(0.3, index=close.columns)
    dear = cheap.copy()
    dear["a"] = 3.0
    assert cost(steady, session, dear, years) > cost(steady, session, cheap, years)
    assert np.isclose(cost(steady, session, cheap, years), 0.5 * 2 * days * 0.3 / 1e4 / years)

    # The rotation keeps the exposure it is a null for: same absolute weight, different phase.
    held = pos.to_numpy()
    rolled = np.roll(held[:, 0], 7)
    assert np.isclose(np.abs(rolled).sum(), np.abs(held[:, 0]).sum()), "a roll holds the same book"


if __name__ == "__main__":
    import sys

    if "--run" in sys.argv:
        from alpacatrader.session import EXTREMA

        built = panel.build()
        fee = fees(built)
        candidates = columns.exhaustion(built, EXTREMA)
        candidates["composite"] = columns.composite(built, EXTREMA)
        rows = []
        for name in ("stretch", "streak", "composite", "volume_climax_seasonal"):
            for sign in (-1, 1):
                rows.append(price(name, candidates[name], built, fee, sign=sign))
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        _selfcheck()
        print("ok — a book that knows the future beats its rotation, one that does not cannot")
