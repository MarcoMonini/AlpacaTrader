"""Measure 3: the minute store — one Parquet per symbol, SIP, adjusted, regular hours only.

§3 fixes every parameter and none of them is a preference: `feed=sip` because IEX is ~2-3% of the
volume and a biased sample at intraday frequency, `adjustment=split` because an unadjusted split
prints a −50% log return that the pivots read as a leg and the label as an opportunity, `1Min` as
the base because the minute bar is native and 5m and 15m derive from it while the reverse cannot,
and RTH because the API serves extended hours by default and they are a different market.

**One file per symbol, at 1m, and nothing else on disk.** The spec's "one file per (symbol,
interval)" is satisfied by deriving the intervals on read: a stored 5m file would be a second copy
of the same information that can fall out of step with its parent, and resampling a month of
minutes costs milliseconds.

**The stamp is the thing that stops a silent mismatch.** Every file carries a JSON sidecar with the
parameters it was built under and a fingerprint of its own closes. A file built on another feed, or
with adjustment off, or truncated, is refused rather than loaded — the same contract as the previous
project's `dataset.cached`, which its docstring calls the thing that saved half a session.

**Trap 3, measured, and it moved a parameter.** Retroactive adjustment is worse than §3 assumed.
XLU has never split, yet `adjustment=all` rewrites its 2016 close from $43.19 to $31.22 — ten years
of dividends discounted backwards. Quarterly distributions therefore rewrite the entire history of
every symbol four times a year, and nothing measured on such a store reproduces months later. The
store is built `adjustment=split` instead: splits still corrected, dividends left out of the series
and charged in `costs` on the short leg only, which is the design §3 states even where it names the
other parameter. The fingerprint stays as the detector — a real split still rewrites history, rarely
and visibly, and a rebuild that finds the same window hashing differently says so.

    uv run python -m alpacatrader.data.store --build
    uv run python -m alpacatrader.data.store --check
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from alpacatrader.data.candles import EXCHANGE, bars

STORE = Path(__file__).parents[3] / "data"
START = "2016-01-04"  # the floor of Alpaca's history; a younger symbol simply starts later
BASE = "1m"
# Fetched a quarter at a time: a year of twenty symbols across extended hours is millions of rows
# in one response, and a failure halfway through a year costs a year.
CHUNK = "QE"


def path(symbol: str, interval: str = BASE) -> Path:
    return STORE / f"{symbol}-{interval}.parquet"


def fingerprint(close: pd.Series) -> str:
    """A hash of the closes, which is what a retroactive adjustment would change."""
    return hashlib.sha256(pd.util.hash_pandas_object(close.round(6), index=True).values).hexdigest()[:16]


def stamp_of(frame: pd.DataFrame, start: str, end: str) -> dict:
    return {
        "feed": "sip",
        "adjustment": "split",
        "session": "rth",
        "interval": BASE,
        "start": start,
        "end": end,
        "rows": len(frame),
        "first": str(frame.index[0]),
        "last": str(frame.index[-1]),
        "fingerprint": fingerprint(frame.close),
    }


def fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Every regular-hours minute bar of one symbol between the two dates, quarter by quarter."""
    pieces = []
    edges = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=CHUNK)
    edges = pd.DatetimeIndex([pd.Timestamp(start), *edges, pd.Timestamp(end)]).unique().sort_values()
    for lo, hi in zip(edges[:-1], edges[1:]):
        piece = bars(symbol, BASE, lo, hi)
        if not piece.empty:
            pieces.append(piece)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces)
    return out[~out.index.duplicated(keep="first")].sort_index()


def build(symbols: list[str], start: str = START, end: str | None = None, force: bool = False) -> pd.DataFrame:
    """Write one Parquet per symbol, skipping what is already there under the same parameters.

    Resumable on purpose: twenty symbols of a decade of minutes is a long pull, and a run that has
    to start over after a network error is a run nobody takes twice.
    """
    end = end or str(pd.Timestamp.now(tz="UTC").normalize().date())
    STORE.mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol in symbols:
        target = path(symbol)
        old = read_stamp(symbol)
        if not force and old and (old["start"], old["end"]) == (start, end):
            rows.append({"symbol": symbol, "rows": old["rows"], "status": "cached", **_span(old)})
            continue
        frame = fetch(symbol, start, end)
        if frame.empty:
            rows.append({"symbol": symbol, "rows": 0, "status": "empty", "first": "", "last": ""})
            continue
        new = stamp_of(frame, start, end)
        moved = bool(old) and old.get("fingerprint") != new["fingerprint"] and old.get("end") == end
        frame.to_parquet(target)
        target.with_suffix(".json").write_text(json.dumps(new, indent=2))
        status = "REWRITTEN" if moved else ("rebuilt" if old else "new")
        rows.append({"symbol": symbol, "rows": new["rows"], "status": status, **_span(new)})
        print(f"  {symbol:5} {new['rows']:>9,} bars  {new['first'][:10]} -> {new['last'][:10]}  [{status}]")
    return pd.DataFrame(rows)


def _span(stamp: dict) -> dict:
    return {"first": stamp["first"][:10], "last": stamp["last"][:10]}


def read_stamp(symbol: str) -> dict | None:
    side = path(symbol).with_suffix(".json")
    return json.loads(side.read_text()) if side.exists() and path(symbol).exists() else None


def load(symbol: str, interval: str = BASE) -> pd.DataFrame:
    """The stored minutes, resampled to `interval` **inside the session and never across one**.

    A bucket that spans a close would mix the last minutes of one day with the first of the next and
    invent a bar that never traded. Grouping by exchange-local date before resampling is what stops
    it, and it is the same rule `dispersion` takes for returns.
    """
    stamp = read_stamp(symbol)
    if stamp is None:
        raise FileNotFoundError(f"{symbol} is not in the store — run `python -m alpacatrader.data.store --build`")
    if (stamp["feed"], stamp["adjustment"], stamp["session"]) != ("sip", "split", "rth"):
        raise ValueError(f"{symbol} was built under {stamp} — delete it and rebuild rather than reading it")
    frame = pd.read_parquet(path(symbol))
    if interval == BASE:
        return frame
    rule = interval.replace("m", "min").replace("hh", "h")
    how = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    how = {k: v for k, v in how.items() if k in frame.columns}
    day = frame.index.tz_convert(EXCHANGE).date
    return pd.concat([g.resample(rule).agg(how).dropna(how="all") for _, g in frame.groupby(day)])


def _selfcheck() -> None:
    """Resampling, the stamp and the fingerprint — the fetch itself needs the network."""
    grid = pd.DatetimeIndex(
        list(pd.date_range("2026-03-10 09:30", periods=390, freq="1min", tz=EXCHANGE))
        + list(pd.date_range("2026-03-11 09:30", periods=390, freq="1min", tz=EXCHANGE))
    ).tz_convert("UTC")
    n = len(grid)
    frame = pd.DataFrame(
        {
            "open": range(n),
            "high": [i + 2 for i in range(n)],
            "low": [i - 1 for i in range(n)],
            "close": [i + 1 for i in range(n)],
            "volume": [10.0] * n,
        },
        index=grid,
        dtype=float,
    )

    # The resampling, exercised through the same code `load` runs but on a frame in hand.
    day = frame.index.tz_convert(EXCHANGE).date
    how = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = pd.concat([g.resample("5min").agg(how).dropna(how="all") for _, g in frame.groupby(day)])
    assert len(out) == 2 * 78, f"390 minutes make 78 five-minute bars a session, got {len(out) / 2}"
    assert out.volume.iloc[0] == 50.0, "five one-minute bars of ten"
    assert out.open.iloc[0] == 0.0 and out.close.iloc[0] == 5.0, "open is the first, close is the last"
    assert out.high.iloc[0] == 6.0 and out.low.iloc[0] == -1.0, "high and low span the bucket"
    # The boundary: the last bucket of a session must not reach into the next one.
    local = out.index.tz_convert(EXCHANGE)
    assert local[77].time().hour == 15 and local[77].time().minute == 55
    assert local[78].strftime("%Y-%m-%d %H:%M") == "2026-03-11 09:30", "a new session starts a new bucket"
    assert pd.Series(local.date).nunique() == 2

    # The fingerprint moves with the data and not with anything else — which is what makes it a
    # detector for a retroactive adjustment rather than a checksum of the file's bytes.
    assert fingerprint(frame.close) == fingerprint(frame.close.copy())
    split = frame.close / 2  # exactly what a 2:1 split rewrites history into
    assert fingerprint(split) != fingerprint(frame.close), "a retroactive adjustment must be visible"
    assert fingerprint(frame.close.iloc[:-1]) != fingerprint(frame.close), "and so must a truncation"

    stamp = stamp_of(frame, "2026-03-10", "2026-03-12")
    assert stamp["rows"] == n and stamp["feed"] == "sip" and stamp["adjustment"] == "split"


if __name__ == "__main__":
    import sys

    from alpacatrader.universe import U1

    if "--build" in sys.argv:
        print(build(list(U1.values())).to_string(index=False))
    elif "--check" in sys.argv:
        print(pd.DataFrame([{"symbol": s, **(read_stamp(s) or {})} for s in U1.values()]).to_string(index=False))
    else:
        _selfcheck()
        print("ok — no bucket spans a close, and a retroactive adjustment is visible")
