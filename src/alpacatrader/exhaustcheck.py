"""Measure 5, first half: does any column lead the price, or do they all only summarise it?

The question the previous project answered too late. Everything it measured predicted the *label*:
its trained model reached Rank IC 0.4114 against `swing_leg_target` and **−0.0405 against the
forward return**, with the buy decile significantly negative at 24 and 72 bars. The stated reason
is that the label is largely a clock, so a model fitted to it learns where the bar sits in a leg,
which is arithmetic on the past.

**Why no model.** Fitting one first confounds two questions: a flat result could be the columns or
the fit. A univariate Rank IC per column cannot be blamed on an architecture, a threshold or a fee,
which is the same reason `legcheck` has no rule inside it. A model is worth building only for a
column that already leads on its own.

**The bar to clear.** Not zero. A column has to carry a Rank IC whose **sign is the same on all
four folds** — on twenty symbols a single cross-section has a standard error near 0.24, so a |Rank
IC| of 0.02 that flips sign between folds is noise with a decimal point. The folds are four
different regimes here, which makes sign stability a much harder test than it was on 3.7 years of
one market.

**Two honest limits, stated before the output.** These are univariate reads, so a column that only
works in combination looks flat; and the target is the plain forward return, not one conditioned on
being near a turn, which is the regime the exhaustion columns are designed for. Both are reasons a
negative result here is weaker evidence than a positive one would be.

    uv run python -m alpacatrader.exhaustcheck --run
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpacatrader import metrics, panel
from alpacatrader.session import EXTREMA


def scan(p: dict, candidates: dict[str, pd.DataFrame], horizons=panel.HORIZONS) -> pd.DataFrame:
    """One row per column, horizon and fold: the cross-sectional IC against the forward return.

    Both sides are masked to `usable` first, so a column is never scored on a bar the sample does
    not contain — the warm-up at the head and the unclosed leg at the tail are not evidence.
    """
    keep = p["usable"].to_numpy()
    masks = panel.folds(p)
    rows = []
    for horizon in horizons:
        ahead = panel.forward(p, horizon)
        for name, column in candidates.items():
            for fold, mask in enumerate(masks, start=1):
                take = keep & mask.to_numpy()
                out = metrics.signal(column[take], ahead[take], horizon=horizon, session=p["session"][take])
                rows.append({"column": name, "horizon": horizon, "fold": fold, **out})
    return pd.DataFrame(rows)


def summarise(scanned: pd.DataFrame) -> pd.DataFrame:
    """Mean and dispersion across folds, with the only flag that matters: does the sign hold?

    `stable` is true when every fold agrees on the direction. Without it a mean Rank IC is an
    average of four numbers that may not be measuring the same thing, and the project's own rule is
    that a comparison without a dispersion is not a comparison.
    """
    grouped = scanned.groupby(["column", "horizon"])
    out = pd.DataFrame(
        {
            "rank_ic": grouped.rank_ic.mean().round(4),
            "sd": grouped.rank_ic.std().round(4),
            "t": grouped.rank_ic_t.mean().round(2),
            "breadth": grouped.breadth.mean().round(2),
            "stable": grouped.rank_ic.apply(lambda s: bool(abs(np.sign(s).sum()) == len(s))),
        }
    )
    out["ratio"] = (out.rank_ic.abs() / out.sd.replace(0, np.nan)).round(2)
    return out.sort_values("ratio", ascending=False)


def _selfcheck() -> None:
    """A planted column and a noise column, on a panel built here — no store, no network."""
    per, days, symbols = 60, 24, 5
    rng = np.random.default_rng(0)
    n = per * days
    session = pd.Series(np.repeat(np.arange(days), per))
    bar = pd.Series(np.tile(np.arange(per), days))
    ret = rng.normal(scale=1e-3, size=(n, symbols))
    close = pd.DataFrame(np.exp(np.cumsum(ret, axis=0)), columns=list("abcde"))
    p = {
        "close": close,
        "session": session,
        "bar": bar,
        "size": session.map(session.value_counts()),
        "usable": pd.Series(np.ones(n, dtype=bool)),
    }

    ahead = panel.forward(p, 5)
    planted = ahead + rng.normal(scale=1e-4, size=ahead.shape)  # knows the answer, plus a little noise
    noise = pd.DataFrame(rng.normal(size=(n, symbols)), columns=close.columns)
    scanned = scan(p, {"planted": planted, "noise": noise}, horizons=(5,))
    assert len(scanned) == 2 * panel.FOLDS

    out = summarise(scanned)
    assert out.loc[("planted", 5), "rank_ic"] > 0.9, "a column that knows the future must show it"
    assert bool(out.loc[("planted", 5), "stable"]), "and show it on every fold"
    assert abs(out.loc[("noise", 5), "rank_ic"]) < 0.05, "and noise must not"
    # The ordering is what a reader acts on: signal above noise, by the ratio to its own dispersion.
    assert out.index[0] == ("planted", 5)

    # A column that is the *negative* of the answer is just as findable, and stays flagged stable —
    # the flag is about agreement between folds, not about the sign being positive.
    flipped = summarise(scan(p, {"flipped": -planted}, horizons=(5,)))
    assert flipped.loc[("flipped", 5), "rank_ic"] < -0.9 and bool(flipped.loc[("flipped", 5), "stable"])

    # A column that leads in two folds and lags in the other two averages near zero *and* is caught
    # by `stable` — which is the failure mode the flag exists for.
    masks = panel.folds(p)
    half = planted.copy()
    for mask in masks[2:]:
        half[mask.to_numpy()] = -planted[mask.to_numpy()]
    mixed = summarise(scan(p, {"mixed": half}, horizons=(5,)))
    assert not bool(mixed.loc[("mixed", 5), "stable"]), "a sign that flips between folds is not a signal"


if __name__ == "__main__":
    import sys

    if "--run" in sys.argv:
        from alpacatrader import columns

        built = panel.build()
        candidates = columns.exhaustion(built, EXTREMA)
        candidates["composite"] = columns.composite(built, EXTREMA)
        table = summarise(scan(built, candidates))
        print(table.to_string())
    else:
        _selfcheck()
        print("ok — a planted column is found, and a sign that flips is not")
