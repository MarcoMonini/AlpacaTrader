"""The dashboard: historical Alpaca bars for one symbol, as candlesticks, with the oracle's pivots.

The deployed artefact of the project and, for now, its whole surface: it proves the environment
reaches the venue, that credentials and feed are wired, and that a session-aware index draws
correctly. Research modules will land beside it as `python -m` entry points, not inside it.

Run it with the `dashboard` config in `.claude/launch.json`, or:

    uv run streamlit run src/alpacatrader/app/dashboard.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alpacatrader import oracle
from alpacatrader.costs import per_side_bp
from alpacatrader.data.candles import CLOSE, EXCHANGE, OPEN, TIMEFRAMES, get_candles
from alpacatrader.universe import U1

HIGH, LOW = "#e74c3c", "#2ecc71"  # the previous project's colours, so a reader is not retrained

# Cached because a rerun fires on every widget touch and each one would otherwise be a download.
# Five minutes: long enough that moving a slider is free, short enough that the last bar is fresh.
load_candles = st.cache_data(ttl=300, show_spinner="Downloading from Alpaca…")(get_candles)


@st.cache_data(show_spinner=False)
def load_legs(df, window: int, lag: int, fee: float):
    """The oracle's trades on what is on screen — the same function the sweep totals.

    Cached on the frame, so moving the window slider costs a pivot pass and not a download. The
    session index is rebuilt from the timestamps, which is what lets this work on whatever timeframe
    the page is showing rather than only on the research grid.
    """
    session = pd.Series(df.index.tz_convert(EXCHANGE).date, index=df.index).factorize()[0]
    return oracle.legs(df.close.reset_index(drop=True), session, window, fee, lag), session


def pivots_on(fig, df, piv, took=None) -> None:
    """Hollow squares for the pivots, solid triangles for the fills — and the two are not the same.

    A square is where the swing turned. A triangle is where the trade actually happened, which with
    the detection lag charged is `window` bars later, at a price the turn has already left. Drawing
    only the squares is what makes a gross of +0.1% look like an arithmetic error next to swings of
    a percent and a half: the eye measures square to square and the number measures triangle to
    triangle. Hollow and solid, so the distinction survives a glance.
    """
    for kind, colour in ((1, HIGH), (-1, LOW)):
        side = piv[piv.kind == kind]
        fig.add_trace(
            go.Scatter(
                x=df.index[side.row.to_numpy()],
                y=side.price,
                mode="markers",
                marker=dict(size=6, color=colour, symbol="square-open", line=dict(width=1.2, color=colour)),
                name="high" if kind == 1 else "low",
                showlegend=False,
                hovertemplate="%{x}<br>pivot %{y}<extra></extra>",
            )
        )
    if took is None or not len(took):
        return
    for rows, prices, colour, mark, label in (
        (took.entry, took.buy, LOW, "triangle-up", "buy"),
        (took.exit, took.sell, HIGH, "triangle-down", "sell"),
    ):
        fig.add_trace(
            go.Scatter(
                x=df.index[rows.to_numpy()],
                y=prices,
                mode="markers",
                marker=dict(size=9, color=colour, symbol=mark, line=dict(width=0.5, color="#111")),
                name=label,
                showlegend=False,
                hovertemplate="%{x}<br>" + label + " %{y}<extra></extra>",
            )
        )


def chart(df, symbol: str, timeframe: str, piv=None, took=None) -> go.Figure:
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
    if piv is not None and len(piv):
        pivots_on(fig, local, piv, took)
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

    # The exposure next to the ticker: XHB is a decision about homebuilders, and a list of twenty
    # tickers alone does not say that anywhere.
    exposures = {instrument: name for name, instrument in U1.items()}

    show_pivots = st.sidebar.checkbox("Oracle pivots", value=True)
    window = st.sidebar.slider(
        "Extrema window (bars)",
        2,
        30,
        5,
        disabled=not show_pivots,
        help="the centred window a pivot is the extreme of. A pivot is not knowable until `window` "
        "bars after it happened, which is what the lag below charges.",
    )
    lagged = st.sidebar.checkbox(
        "Charge the detection lag",
        value=True,
        disabled=not show_pivots,
        help="fill `window` bars after the pivot — the earliest a causal reader could act. "
        "Unticked is perfect hindsight, which is a ceiling and not a strategy.",
    )

    symbol = st.sidebar.selectbox(
        "Symbol",
        list(U1.values()),
        format_func=lambda s: f"{s} · {exposures[s]}",
        accept_new_options=True,
        help="U1 — one instrument per exposure, chosen on decorrelation and then on measured "
        "spread. Type any other US ticker to draw it.",
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

    piv = took = trades = perfect = None
    if show_pivots:
        fee = per_side_bp(float(df.close.iloc[-1]), 0.01) / 1e4
        trades, session = load_legs(df, window, window if lagged else 0, fee)
        piv = oracle.find(df.close.reset_index(drop=True), session, window)
        took = oracle.tradable(trades)
        # The other reading, always, because the whole point of the oracle is the gap between them.
        perfect = oracle.tradable(load_legs(df, window, 0, fee)[0])

    st.plotly_chart(chart(df, symbol, timeframe, piv, took), use_container_width=True)
    st.caption(f"{len(df):,} bars · {df.index[0]:%Y-%m-%d %H:%M} → {df.index[-1]:%Y-%m-%d %H:%M} UTC")

    if took is not None:
        sessions = len(np.unique(session))
        years = sessions / 252
        columns = st.columns(7)
        columns[0].metric("Legs traded", f"{len(took):,}", help="lows closed by the next high, inside one session")
        columns[1].metric(
            "Hindsight gross",
            f"{100 * perfect.gross.sum():+.1f}%" if len(perfect) else "—",
            help="square to square: buying every low and selling every high. Unreachable by "
            "construction — a pivot is not knowable until `window` bars after it happened.",
        )
        columns[2].metric(
            "Gross total",
            f"{100 * took.gross.sum():+.1f}%" if len(took) else "—",
            delta=(
                f"{100 * (took.gross.sum() - perfect.gross.sum()):+.1f}% vs hindsight"
                if len(took) and len(perfect)
                else None
            ),
            delta_color="off",
            help="triangle to triangle: what the fills actually earned. With the lag charged this "
            "is a different number from the one above, and the gap is what the lag costs.",
        )
        columns[3].metric(
            "Net total",
            f"{100 * took.net.sum():+.1f}%" if len(took) else "—",
            help=f"after {1e4 * fee:.2f} bp a side, charged twice a leg",
        )
        columns[4].metric(
            "Log / year",
            f"{float(np.log1p(took.net).sum() / years):+.3f}" if len(took) and years else "—",
            help="what the sweep totals: net log P&L annualised over the sessions on screen",
        )
        columns[5].metric(
            "Crossed the bell",
            f"{int(trades.crosses.sum())}",
            help="legs spanning a close — counted, never traded, because the book is flat at 16:00",
        )
        columns[6].metric("Median bars", f"{took.bars.median():.0f}" if len(took) else "—")
        if lagged and len(perfect):
            st.caption(
                f"Squares are the {len(perfect):,} pivots hindsight would have traded "
                f"({100 * perfect.gross.sum():+.1f}%); triangles are the {len(took):,} fills a causal "
                f"reader could reach, {window} bars later ({100 * took.gross.sum():+.1f}%). "
                "The distance between a square and the triangle beside it is what the lag costs."
            )
        with st.expander(f"The {len(took):,} legs, as counted"):
            shown = took.assign(
                opened=df.index[took.entry].tz_convert(EXCHANGE),
                closed=df.index[took.exit].tz_convert(EXCHANGE),
                gross=(100 * took.gross).round(3),
                net=(100 * took.net).round(3),
            )[["opened", "closed", "buy", "sell", "gross", "net", "bars"]]
            st.dataframe(shown, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
