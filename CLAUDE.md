# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The equity half of the study that ran on crypto in `TradingVision`: the same machinery — centred
pivots, a leg label, a multi-branch GRU, exact purging, policy optimisation with the fee inside the
reward — moved from twenty USDT pairs to a basket of US ETFs and stocks on Alpaca, where commission
is zero and execution cost falls by a factor of 15 to 150.

Right now the repository is the environment and nothing else: the data layer and the dashboard.
None of the research modules have been written, and **no P&L has ever been measured on equity**.
Crypto numbers are quotable only to say which cost threshold the previous project failed to clear.

`equity_dataset_schema.html` (Italian) is the spec and the lab notebook: it inherits every closed
decision of the crypto project and marks it *riusabile*, *da rimisurare*, or *decaduto*. **Read it
before writing any research module** — most of what looks like a fresh decision is in there with
the reasoning that settled it, and several crypto constants are explicitly flagged as needing to be
re-measured rather than carried over. Keep it current when a step lands; the git history should read
as a sequence of measurements, and commit subjects are written that way.

## Commands

```bash
uv sync                                          # dev group included
uv run pytest -q
uv run ruff check . && uv run black --check .    # what CI runs, line-length 120
uv run python -m alpacatrader.data.candles       # a module's self-check, run directly
```

The dashboard: `preview_start` with the `dashboard` config in `.claude/launch.json`, or
`uv run streamlit run src/alpacatrader/app/dashboard.py`.

## Credentials

Two environment variables, Alpaca's own names, in `.env` (gitignored — copy `.env.example`):
`APCA_API_KEY_ID` and `APCA_API_SECRET_KEY`. Paper-account keys are enough; nothing here places an
order. `ALPACA_FEED` picks the feed and defaults to `iex`.

`alpaca-py` does *not* read those variables itself — it raises on a missing key — so
`data.candles.client()` reads them, and reads them **lazily**. A module that touched credentials at
import would take Streamlit down before it drew the error message that says which variable is
missing.

## Architecture

**The feed choice is not cosmetic.** `iex` is what the free plan serves and is roughly 2–3% of the
volume — a biased sample for anything at intraday frequency. `sip` is the consolidated tape, needs a
market-data subscription, and is the feed every number in the spec is to be measured on. Bars are
requested with `adjustment=all` (splits and dividends); the spec notes the dividend adjustment has
to be discounted again in the cost of the short side.

**Equity time is not continuous, and that is the assumption the crypto project cannot lend us.**
There, a bucket with no trade got a synthetic flat bar, because the market never closes. Here,
between the 15:55 bar and the 09:30 bar of the next day pass 17.5 hours in which nothing could have
traded. So `data.candles` fills nothing: the holes are real and they stay. The dashboard closes
them on the axis with Plotly `rangebreaks`; what closes them inside a model — the branch alignment
rule `label + tf - 5m`, which assumed one bar duration between adjacent bars — is section 7 of the
spec and is **open**. Do not reuse the crypto alignment code without reading it.

**Sessions are compared on the exchange's clock.** `regular_hours` converts to `America/New_York`
before comparing against 09:30/16:00, because the UTC offset moves twice a year and the session does
not. Half-days need no special case (nothing prints after an early close) and a market holiday is
simply a day with no bars. Daily bars skip the filter entirely — Alpaca stamps them at midnight, so
filtering them by open time would drop every one.

**Every frame is indexed by the *open* time of its bar, in UTC.** A bar labelled `b` on timeframe
`tf` closes at `b + tf`. One bar of anticipation on a slower branch hands a model a window of future
and inflates every metric downstream. This is the one rule that must never break; the crypto project
enforced it with a truncation test and this one will need its own.

**The universe is not settled.** `SYMBOLS` in `data.candles` is a starting point for the page, not
the twenty the spec will measure on: section 4 leaves the choice open on a spread measurement nobody
has taken yet, and has a decision rule written down *before* the output is looked at. Don't quietly
promote the page's list into a research module.

## Conventions

Self-checks live at the bottom of each module as asserts under `if __name__ == "__main__"` (or a
`_selfcheck()` function when the `__main__` is the real run), not in a mirrored test file.
`tests/test_selfchecks.py` is what makes CI run them — **add every new module to its list**.

Module docstrings carry the reasoning: what the module measures, which numbers came out, and why an
alternative was rejected. That is where the project's memory lives, so keep them accurate rather
than short. Comments explain the decision, not the syntax.

Constants are measured, not assumed, and say so where they are defined. A constant inherited from
the crypto project is a constant to re-measure unless the spec marks it *riusabile*.

Research modules are `python -m` entry points under `src/alpacatrader/`, each one a stage that has
to beat the previous one out of sample. The dashboard is the deployed artefact; everything else runs
by hand.
