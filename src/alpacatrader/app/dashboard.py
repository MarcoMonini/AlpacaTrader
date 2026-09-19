"""The dashboard: historical Alpaca bars for one symbol, as candlesticks.

The deployed artefact of the project and, for now, its whole surface: it proves the environment
reaches the venue, that credentials and feed are wired, and that a session-aware index draws
correctly. Research modules will land beside it as `python -m` entry points, not inside it.

Run it with the `dashboard` config in `.claude/launch.json`, or:

    uv run streamlit run src/alpacatrader/app/dashboard.py
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from alpacatrader.data.candles import CLOSE, EXCHANGE, OPEN, SYMBOLS, TIMEFRAMES, get_candles

# Cached because a rerun fires on every widget touch and each one would otherwise be a download.
# Five minutes: long enough that moving a slider is free, short enough that the last bar is fresh.
load_candles = st.cache_data(ttl=300, show_spinner="Downloading from Alpaca…")(get_candles)


def chart(df, symbol: str, timeframe: str) -> go.Figure:
    """Candles over volume, with the hours the market was shut removed from the axis.

    Without `rangebreaks` a 5m chart is mostly empty space: every night is 17.5 hours of axis with
    no bar on it and every weekend is two days, so the candles compress into thin vertical strips.
    The breaks are cut on the exchange's own clock — Plotly evaluates them in the axis timezone,
    which is why the index is converted rather than left in UTC.
    """
    local = df.tz_convert(EXCHANGE)
    fig = go.Figure(
        [
            go.Candlestick(
                x=local.index, open=local.open, high=local.high, low=local.low, close=local.close, name=symbol
            ),
            # Its own axis at the bottom: volume in shares and price in dollars share no scale, and
            # putting them on one flattens the candles onto the top edge.
            go.Bar(x=local.index, y=local.volume, name="volume", marker_color="#888", opacity=0.4, yaxis="y2"),
        ]
    )
    breaks = [dict(bounds=["sat", "mon"])]
    if timeframe != "1d":
        breaks.append(dict(bounds=[CLOSE.hour, OPEN.hour + OPEN.minute / 60], pattern="hour"))
    fig.update_layout(
        height=640,
        margin=dict(t=30, b=10),
        showlegend=False,
        xaxis_rangeslider_visible=False,
        xaxis=dict(rangebreaks=breaks),
        yaxis=dict(domain=[0.25, 1.0], title="price"),
        yaxis2=dict(domain=[0.0, 0.2], title="vol"),
        # Keeps zoom and pan across reruns: Plotly patches the existing figure instead of
        # remounting it. The value changes with what was fetched, so the view resets when the
        # symbol or the period does and survives everything else.
        uirevision=f"{symbol}-{timeframe}-{len(df)}",
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="Alpaca Trader", layout="wide")
    st.title("Alpaca Trader")

    symbol = st.sidebar.selectbox(
        "Symbol",
        SYMBOLS,
        accept_new_options=True,
        help="a starting universe, not the one the spec settles on. Type any other US ticker to draw it.",
    )
    timeframe = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES), index=2)
    # Calendar days and not sessions, because that is the window the request is actually made on.
    days = st.sidebar.slider("History (days)", 1, 365, 30)
    rth = st.sidebar.checkbox(
        "Regular hours only",
        value=True,
        help="pre- and post-market are a few percent of the volume at much wider spreads — a "
        "different market, not an extension of this one.",
    )

    try:
        df = load_candles(symbol, timeframe, days, rth)
    except Exception as error:  # credentials, an unknown ticker, a feed the plan lacks
        st.error(str(error))
        return
    if df.empty:
        st.warning(f"Alpaca serves no {timeframe} bars for {symbol} over the last {days} days.")
        return

    st.plotly_chart(chart(df, symbol, timeframe), use_container_width=True)
    st.caption(f"{len(df):,} bars · {df.index[0]:%Y-%m-%d %H:%M} → {df.index[-1]:%Y-%m-%d %H:%M} UTC")


if __name__ == "__main__":
    main()
