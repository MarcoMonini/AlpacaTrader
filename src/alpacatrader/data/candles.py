"""Equity OHLCV from Alpaca.

Unlike the crypto endpoint of the previous project, the stock one needs credentials: set
`APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` (Alpaca's own names, so every other tool in the
ecosystem reads the same two). Paper-account keys are enough — nothing here places an order.

`ALPACA_FEED` picks the feed and defaults to `sip`, the consolidated tape. Measured 2026-09-19: the
historical SIP endpoint answers on free paper keys back to 2016-01-04, and only the *real-time*
feed is paid — so there is no reason to train on anything else. IEX is ~2-3% of the volume (SPY 1m
at 2024-06-03 14:30 reads v=577 / n=14 against SIP's v=75,038 / n=2,495) and its daily history
starts in 2018 with gaps, which is what made the first run of `universe` unusable.

Bars come back split- and dividend-adjusted (`adjustment=all`) and indexed by the *open* time of
the bar in UTC, which is the alignment rule everything downstream depends on.

**No gapless grid here, unlike the crypto project.** There, a period with no trade got a synthetic
flat bar because the market never closes. An equity session does: between the 15:55 bar and the
09:30 bar of the next day pass 17.5 hours in which nothing could have traded, and filling them
would invent thousands of bars a day. The holes are real and stay; what closes them on screen is
`plotly`'s `rangebreaks`, and what closes them in a model is a decision the spec has not taken yet.
"""

from __future__ import annotations

import os
from datetime import datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

TIMEFRAMES = {
    "1m": TimeFrame(1, TimeFrameUnit.Minute),
    "5m": TimeFrame(5, TimeFrameUnit.Minute),
    "15m": TimeFrame(15, TimeFrameUnit.Minute),
    "1h": TimeFrame(1, TimeFrameUnit.Hour),
    "1d": TimeFrame(1, TimeFrameUnit.Day),
}

# The exchange's own clock. Every session boundary is a wall-clock time in New York, so it is the
# only timezone in which 09:30 means 09:30 on both sides of the two DST changes a year.
EXCHANGE = "America/New_York"
OPEN, CLOSE = time(9, 30), time(16, 0)

# No symbol list lives here. This module fetches whatever it is asked for, and the universe is a
# measured decision that belongs to `universe` (E1, then U1) — which reads *this* module to take it.
# A list here would either duplicate that decision or invert the dependency.

# The repository root, from this file rather than from the working directory: the page is started
# by an absolute path as often as not, and `.env` does not move when the caller does.
ENV_FILE = Path(__file__).parents[3] / ".env"

_client: StockHistoricalDataClient | None = None


def load_env(path: Path = ENV_FILE) -> None:
    """Put `KEY=value` lines from `.env` into the environment, without overwriting what is there.

    Five lines instead of a dependency: this reads two secrets out of a file the developer wrote by
    hand, and `python-dotenv`'s interpolation, export syntax and multi-line values are features
    nothing here uses. A missing file is the normal case in a container, where the variables are
    injected by the host.

    `setdefault` and not assignment, so a variable already exported wins over the file. That is the
    order a deployment needs — the image carries no `.env` and the host's variables must not be
    shadowed by a stale one that got copied in by mistake.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not key.lstrip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip())


def client() -> StockHistoricalDataClient:
    """The shared client, built on first use.

    Lazily and not at import: the page has to be able to draw its own error message when the two
    variables are missing, and a module that raises at import takes Streamlit down before it
    renders anything.
    """
    global _client
    if _client is None:
        load_env()
        key, secret = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if not (key and secret):
            raise RuntimeError("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY — see .env.example.")
        _client = StockHistoricalDataClient(key, secret)
    return _client


def regular_hours(bars: pd.DataFrame) -> pd.DataFrame:
    """Only the bars opening inside the regular session, 09:30 to 16:00 New York.

    Alpaca serves pre- and post-market minutes on the same endpoint, and they are a different
    market: a few percent of the volume, wider spreads, and prints that no cost figure in the spec
    was measured against. Mixing them into an intraday series silently changes what a window of
    N bars spans.

    Compared on the exchange's clock rather than on a UTC offset, because the offset moves twice a
    year and the session does not. Half-days (13:00 closes) need no special case — they simply
    print no bar after their close — while a market holiday is a day with no bars at all.
    """
    local = bars.index.tz_convert(EXCHANGE)
    return bars[(local.time >= OPEN) & (local.time < CLOSE) & (local.dayofweek < 5)]


def get_candles(symbol: str, timeframe: str, days: int, rth: bool = True) -> pd.DataFrame:
    """OHLCV of one symbol over the last `days` calendar days, indexed by UTC bar-open time.

    Empty DataFrame when Alpaca serves nothing for the symbol. `days` counts calendar days and not
    sessions, so a 5-day window over a weekend is three sessions — the honest reading, since it is
    the wall clock the data is requested on.
    """
    bars = (
        client()
        .get_stock_bars(
            StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TIMEFRAMES[timeframe],
                start=datetime.now(timezone.utc) - timedelta(days=days),
                adjustment=Adjustment.ALL,
                feed=DataFeed(os.environ.get("ALPACA_FEED", "sip")),
            )
        )
        .df
    )
    if bars.empty:
        return bars
    # The index is (symbol, timestamp): with a single symbol the first level is noise.
    bars = bars.droplevel("symbol").sort_index()
    # Daily bars are already one per session; filtering them by open time would drop every one,
    # since Alpaca stamps them at midnight.
    return regular_hours(bars) if rth and timeframe != "1d" else bars


def _selfcheck() -> None:
    """The session filter, on an index built by hand — the fetch itself needs the network."""
    # A 1m grid across a whole day, from before the open to after the close, on both sides of the
    # spring DST change: in March the session is 13:30 UTC, in January 14:30, and a filter written
    # against a fixed offset gets exactly one of the two right.
    for day, offset in (("2025-01-15", 5), ("2025-03-12", 4)):
        index = pd.date_range(f"{day} 08:00", f"{day} 23:59", freq="1min", tz="UTC")
        bars = pd.DataFrame({"close": 1.0}, index=index)
        kept = regular_hours(bars).index.tz_convert(EXCHANGE)
        assert len(kept) == 390, f"{day}: a regular session is 390 one-minute bars, got {len(kept)}"
        assert kept[0].time() == OPEN and kept[-1].time() == time(15, 59)
        assert kept[0] == pd.Timestamp(f"{day} {9 + offset}:30", tz="UTC"), "the offset moves, the session does not"

    # A Saturday is not a session, whatever the clock says.
    weekend = pd.date_range("2025-01-18 14:30", periods=60, freq="1min", tz="UTC")
    assert regular_hours(pd.DataFrame({"close": 1.0}, index=weekend)).empty

    # Half-days need no rule of their own: nothing prints after the early close, so nothing is cut.
    early = pd.date_range("2025-07-03 13:30", "2025-07-03 17:00", freq="1min", tz="UTC")
    assert len(regular_hours(pd.DataFrame({"close": 1.0}, index=early))) == len(early)

    # `.env` parsing, on a file written here rather than on the developer's own.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        env = Path(tmp) / ".env"
        env.write_text("# a comment\nAPCA_TEST_KEY=abc\nAPCA_TEST_TAKEN = already \nblank line\n")
        os.environ["APCA_TEST_TAKEN"] = "exported"
        try:
            load_env(env)
            assert os.environ["APCA_TEST_KEY"] == "abc", "a key=value line reaches the environment"
            assert os.environ["APCA_TEST_TAKEN"] == "exported", "and an exported variable wins over the file"
        finally:
            del os.environ["APCA_TEST_KEY"], os.environ["APCA_TEST_TAKEN"]
    load_env(Path(tmp) / "absent")  # a missing file is the container's normal case, not an error

    assert not os.environ.get("APCA_API_KEY_ID") or client(), "credentials, when present, build a client"


if __name__ == "__main__":
    _selfcheck()
    print("ok — the session filter holds across DST")
