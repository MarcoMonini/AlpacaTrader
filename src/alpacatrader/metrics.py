"""The qlib signal set, cross-sectionally, on wide frames.

Ported unchanged in substance from the previous project, where §8 marks it *riusabile*: twenty
symbols here as there, so the standard error of one cross-section is still ~1/√17 ≈ 0.24 and a
single date's IC is still nearly pure noise. Only the average over thousands of instants means
anything, and a comparison without a dispersion across folds is not a comparison.

**Wide and not long.** The previous project carried a `(timestamp, symbol)` MultiIndex because its
panel was ragged. Here every symbol sits on one session grid, so a frame of rows by symbols says
the same thing, computes the cross-section as a row operation, and makes a missing symbol a NaN in
a row rather than an absent key.

**`spearman` instead of `Series.corr(method="spearman")`, and this is not style.** pandas imports
scipy *lazily*, from inside `pandas.core.nanops`, so the call survives every import-time check and
every test run in a venv that happens to have it, then raises on the host that serves the page. It
did exactly that once. Pearson on the ranks is Spearman, and `rank()` and the default `corr()` are
numpy alone.

**The raw ICIR is not a significance when the label overlaps.** At a 30-bar forward return on a
3-bar grid, adjacent instants share 29 of the 30 bars their labels are made of, so the dispersion
in the ICIR's denominator is that of a 30-point moving average and the ratio reads several times
too high. `blocked` collapses onto non-overlapping blocks — **counted in bars and inside a session**,
because on session time there is no clock to floor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_BREADTH = 3  # a correlation over two points is 1 or -1 and says nothing


def by_bar(pred: pd.DataFrame, target: pd.DataFrame, rank: bool = False) -> pd.Series:
    """Correlation inside each row, over the symbols present in both.

    Rows holding fewer than `MIN_BREADTH` symbols come back NaN. Pearson by hand rather than a
    row-apply of `Series.corr`: the whole thing stays vectorised, and on the ranks it is Spearman.
    """
    pred, target = pred.align(target, join="inner")
    both = pred.notna() & target.notna()
    pred, target = pred.where(both), target.where(both)
    if rank:
        # Ranked after the masking and not before: ranking each frame over its own valid cells
        # would rank one of them over symbols the other cannot match, which is a different number.
        pred, target = pred.rank(axis=1), target.rank(axis=1)
    n = both.sum(axis=1)
    zp = pred.sub(pred.mean(axis=1), axis=0).div(pred.std(axis=1), axis=0)
    zy = target.sub(target.mean(axis=1), axis=0).div(target.std(axis=1), axis=0)
    return ((zp * zy).sum(axis=1) / (n - 1)).where(n >= MIN_BREADTH)


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman between two aligned series, *through time* — the shape scipy would be imported for.

    The `dropna` comes before the ranking, which is the part that is easy to get wrong: `corr`
    drops pairs where either side is NaN, so ranking each series over its own valid rows first
    would rank one of them over rows the other cannot match. Ranking the aligned frame is what
    makes this equal to pandas and not merely close to it.
    """
    both = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(both) < MIN_BREADTH:
        return float("nan")
    ranked = both.rank()
    return float(ranked.a.corr(ranked.b))


def blocked(per_bar: pd.Series, horizon: int, session: pd.Series) -> dict[str, float]:
    """`per_bar` collapsed onto non-overlapping blocks of `horizon` bars — the honest denominator.

    Blocks are cut **inside a session**, never across one: a block spanning a close would average
    instants whose forward returns belong to different days, which is the same mistake the session
    index exists to stop everywhere else. Two runs over the same period therefore compare block for
    block whatever rows each happens to hold.
    """
    valid = per_bar.dropna()
    if valid.empty:
        return {"mean": np.nan, "se": np.nan, "t": np.nan, "ir": np.nan, "blocks": 0}
    where = session.reindex(valid.index)
    # Position inside the session, by arrival order, so blocks are cut from each session's own
    # start rather than from a global offset that would straddle every boundary differently.
    within = pd.Series(np.arange(len(valid)), index=valid.index).groupby(where.to_numpy()).rank(method="first") - 1
    block = valid.groupby([where.to_numpy(), (within // horizon).to_numpy()]).mean()
    n = len(block)
    sd = float(block.std())
    se = sd / np.sqrt(n) if n > 1 else np.nan
    return {
        "mean": float(block.mean()),
        "se": float(se),
        "t": float(block.mean() / se) if n > 1 and sd > 0 else np.nan,
        "ir": float(block.mean() / sd) if n > 1 and sd > 0 else np.nan,
        "blocks": n,
    }


def signal(
    pred: pd.DataFrame, target: pd.DataFrame, horizon: int | None = None, session: pd.Series | None = None
) -> dict[str, float]:
    """IC, ICIR, Rank IC, Rank ICIR for one set of predictions, on a shared wide grid.

    With `horizon` and `session`, it adds the three numbers that are readable when adjacent labels
    overlap: `rank_ic_se`, `rank_ic_t` and `blocks`. `rank_ic_t` is the one to read beside a
    difference between two steps; `rank_icir` stays in the output because every number the previous
    project measured was taken on it, but at an overlapping horizon it is not a significance.
    """
    out: dict[str, float] = {}
    for name, rank in (("ic", False), ("rank_ic", True)):
        per = by_bar(pred, target, rank).dropna()
        out[name] = float(per.mean())
        out[f"{name}ir"] = float(per.mean() / per.std()) if per.std() > 0 else np.nan
        if rank and horizon is not None and session is not None:
            b = blocked(per, horizon, session)
            out["rank_ic_se"], out["rank_ic_t"], out["blocks"] = b["se"], b["t"], b["blocks"]
    # Counted over the rows that actually carry a cross-section, not over every row handed in. A
    # caller passing a whole fold at a 30-bar horizon also passes the rows near each close, whose
    # forward return is NaN by design; averaging their zeroes in reported 15.4 where the panel holds
    # 19.46, which §9 would read as a diluted file and reject. The correlation was never affected —
    # `by_bar` drops those rows — so the number was wrong and nothing computed from it was.
    counts = (pred.notna() & target.notna()).sum(axis=1)
    out["breadth"] = float(counts[counts >= MIN_BREADTH].mean())
    return out


def _selfcheck() -> None:
    """Against pandas' own correlations, which need scipy for the ranks — equal, not close."""
    rng = np.random.default_rng(0)
    y = pd.DataFrame(rng.normal(size=(200, 6)), columns=list("abcdef"))

    assert np.isclose(signal(y, y)["ic"], 1.0) and np.isclose(signal(y, y)["rank_ic"], 1.0)
    assert np.isclose(signal(-y, y)["ic"], -1.0)
    # Monotone but not linear: Rank IC stays perfect, IC does not.
    cubed = signal(y**3, y)
    assert np.isclose(cubed["rank_ic"], 1.0) and cubed["ic"] < 0.95
    noise = signal(pd.DataFrame(rng.normal(size=y.shape), columns=y.columns), y)
    assert abs(noise["rank_ic"]) < 0.05 and abs(noise["rank_icir"]) < 1

    # Row by row against pandas, which is the definition this replaces.
    row = y.iloc[0]
    assert np.isclose(by_bar(y**3, y).iloc[0], (row**3).corr(row))
    assert np.isclose(by_bar(y**3, y, rank=True).iloc[0], (row**3).corr(row, method="spearman"))

    # Three kinds of row, and breadth has to treat them differently. A row of four symbols out of
    # six is thin and counts; a row of two has no cross-section at all and is not a thin row but an
    # absent one; a row of none is the same. Averaging the last two in as zeroes is what made a
    # full panel read as a diluted one.
    thin = y.copy()
    thin.iloc[0, 4:] = np.nan  # four symbols: thin, usable, counted
    assert by_bar(thin, y).notna().all(), "four symbols still make a cross-section"
    assert signal(thin, y)["breadth"] < 6.0, "and a thin row pulls the average down"

    absent = y.copy()
    absent.iloc[0, 2:] = np.nan  # two symbols: below MIN_BREADTH, no correlation to be had
    assert np.isnan(by_bar(absent, y).iloc[0]) and by_bar(absent, y).iloc[1:].notna().all()
    assert np.isclose(signal(absent, y)["breadth"], 6.0), "a row without a cross-section is not counted"

    empty = y.copy()
    empty.iloc[:100] = np.nan
    assert np.isclose(signal(empty, y)["breadth"], 6.0), "and neither are a hundred of them"

    # `spearman` equals pandas', including when the two series miss different rows.
    a = pd.Series(rng.normal(size=50))
    b = a * 2 + rng.normal(scale=0.3, size=50)
    assert np.isclose(spearman(a, b), a.corr(b, method="spearman"))
    a2, b2 = a.copy(), b.copy()
    a2.iloc[:5] = np.nan
    b2.iloc[45:] = np.nan
    assert np.isclose(spearman(a2, b2), a2.corr(b2, method="spearman")), "dropna before ranking"

    # --- blocking, which is the part that had to be rewritten for session time -------------------
    # Three sessions of 30 instants. A block of 10 bars must never span two sessions, so 9 blocks
    # come out and not 9 minus the boundaries smeared together.
    per = pd.Series(rng.normal(size=90))
    session = pd.Series(np.repeat([0, 1, 2], 30))
    out = blocked(per, 10, session)
    assert out["blocks"] == 9, f"three sessions of three blocks, got {out['blocks']}"
    # A horizon as long as the session gives one block per session, never a block across two.
    assert blocked(per, 30, session)["blocks"] == 3
    assert blocked(per, 45, session)["blocks"] == 3, "a block cannot be longer than its session"
    # Overlap inflation, the thing this exists to remove: a smooth series has a small dispersion
    # across adjacent instants and an honest one across blocks.
    smooth = pd.Series(np.convolve(rng.normal(size=120), np.ones(30) / 30, mode="valid")[:90])
    assert blocked(smooth, 30, session)["se"] > smooth.sem(), "blocks admit the error the raw sem hides"


if __name__ == "__main__":
    _selfcheck()
    print("ok — equal to pandas, and a block never spans a close")
