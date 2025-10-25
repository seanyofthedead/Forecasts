# Live-Simulated Trading Agent

This repository provides a scaffold for experimenting with intraday trading
strategies in a safe, local-only environment.  It includes a deterministic
momentum strategy, hooks for a Stable Baselines3 reinforcement learning (RL)
policy, and a hybrid approach that combines both signals.  The simulation
engine enforces 2:1 reward-to-risk trade management, records trades, and
summarises performance metrics for both live simulations and historical
backtests.

> ⚠️ **Educational use only** – the code is intended for offline research and
> should not be connected to live brokerage accounts or used to deploy real
> capital without significant hardening and testing.

## Repository structure

```
├── backtesting/
│   └── backtester.py          # Utilities for running historical simulations
├── data/
│   ├── data_feed.py           # Real-time / historical data ingestion helpers
│   └── features.py            # Technical indicator calculations
├── metrics/
│   └── metrics.py             # Trade logging and performance analytics
├── simulation/
│   └── engine.py              # Live-style execution engine with risk controls
├── strategies/
│   ├── hybrid.py              # Combines rule-based and RL signals
│   ├── rl_agent.py            # Wrapper around Stable Baselines3 models
│   └── rule_based.py          # Gap-and-go / pullback momentum rules
├── main.py                    # Command line entry point
└── README.md
```

## Getting started

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install pandas yfinance requests matplotlib stable-baselines3 gym
   ```

   The RL components are optional; if Stable Baselines3 is not installed the
   scaffold still runs, but RL driven strategies will fall back to no-ops.

2. **Configure data access**

   Yahoo Finance via `yfinance` requires no API key.  If you prefer to use
   Alpha Vantage for intraday data, export your API key first:

   ```bash
   export ALPHAVANTAGE_API_KEY="YOUR_KEY"
   ```

3. **Prepare historical data for backtests**

   ```python
   import yfinance as yf

   df = yf.download("AAPL", interval="1m", period="5d")
   df.reset_index().rename(columns={"index": "datetime"}).to_csv("data/aapl.csv", index=False)
   ```

## Usage

Run the CLI with `python main.py` and choose between live simulation and
historical backtesting.

### Backtesting

```bash
python main.py --strategy rule_based --mode backtest --symbol AAPL --datafile data/aapl.csv
```

The command prints trade statistics such as win rate, profit factor, drawdown,
and consecutive losers, allowing you to iterate on rule parameters quickly.

### Live simulation (paper trading)

```bash
python main.py --strategy hybrid --mode live --symbol TSLA
```

The engine streams data (via Yahoo Finance by default), executes simulated
trades, and enforces 2:1 reward-to-risk management automatically.  Stop the
run with `Ctrl+C` to see the performance summary.

## Extending the scaffold

- **Model training** – train a Stable Baselines3 agent separately and save it
  into `models/trained_agent.zip` so the RL strategy can load it.
- **Additional indicators** – add new helpers in `data/features.py` and make
  them available to your strategies.
- **Dashboard** – build a Streamlit or textual UI that consumes the trade log
  to visualise live performance.
- **Portfolio support** – generalise the simulation engine to manage multiple
  concurrent positions across different tickers.

## Disclaimer

This project is a pedagogical scaffold.  It omits crucial concerns required for
production trading systems such as order routing, latency handling,
regulatory/compliance checks, and comprehensive risk controls.  Use it only for
research and simulated experimentation.
