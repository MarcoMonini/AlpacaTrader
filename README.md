<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0B2E1F,100:1B6B4A&height=180&section=header&text=AlpacaTrader&fontSize=48&fontColor=ffffff&desc=Intraday%20swing%20research%20on%20US%20equities%20%7C%20Alpaca%20market%20data&descSize=15&descAlignY=72" />

[![CI](https://github.com/MarcoMonini/AlpacaTrader/actions/workflows/ci.yml/badge.svg)](https://github.com/MarcoMonini/AlpacaTrader/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

<p>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white" />
  <img src="https://img.shields.io/badge/uv-DE5FE9?style=flat-square&logo=uv&logoColor=white" />
  <img src="https://img.shields.io/badge/Ruff-D7FF64?style=flat-square&logo=ruff&logoColor=black" />
  <img src="https://img.shields.io/badge/Black-000000?style=flat-square" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=flat-square&logo=githubactions&logoColor=white" />
</p>

</div>

## 🎯 What this is

The equity half of the study that ran on crypto in
[TradingVision](https://github.com/MarcoMonini/TradingVision) — same machinery, different venue:
zero commission and an execution cost 15–150× lower. The lab notebook is
[`equity_dataset_schema.html`](equity_dataset_schema.html) (Italian), which inherits every closed
decision of the previous project and marks it *reusable*, *to re-measure*, or *dead*.

**Today the repository is the environment and the dashboard.** No research module is written yet
and no P&L has ever been measured on equity.

<div align="center">

| Venue | Feed | Universe | Cost per side | Measured on equity |
|:---:|:---:|:---:|:---:|:---:|
| **Alpaca** | **SIP, 2016 →** | **20 symbols (open)** | **0.17–1.34 bp** | **nothing yet** |

</div>

## 🚀 Quick start

```bash
uv sync
cp .env.example .env     # then put your Alpaca keys in it
uv run streamlit run src/alpacatrader/app/dashboard.py
```

Paper-account keys are enough — the project reads market data and places no orders. Without them
the page still starts and says which variable is missing.

The dashboard downloads historical bars for any US ticker and draws them as candlesticks with
volume underneath, one tab per symbol. Nights and weekends are cut out of the axis, so a 5-minute
chart is candles rather than five vertical strips separated by empty days.

## 🧱 Layout

```
src/alpacatrader/
├── data/candles.py      # Alpaca bars, adjusted, session-filtered
└── app/dashboard.py     # the Streamlit page — the deployed artefact
tests/test_selfchecks.py # runs each module's own asserts
```

Research modules land beside `data/` as `python -m` entry points, each a stage that has to beat the
previous one out of sample.

## 🔬 Commands

```bash
uv run pytest -q
uv run ruff check . && uv run black --check .    # what CI runs, line-length 120
uv run python -m alpacatrader.data.candles       # one module's self-check
docker build -t alpacatrader .                   # the page, containerised
```

## 📐 Conventions

Asserts live at the bottom of the module they check, not in a mirrored test file;
`tests/test_selfchecks.py` is what makes CI run them. Module docstrings carry the reasoning — what
was measured, what came out, why the alternative was rejected. Constants are measured, not assumed,
and a constant inherited from the crypto project is one to re-measure unless the spec says
otherwise. The details are in [`CLAUDE.md`](CLAUDE.md).
