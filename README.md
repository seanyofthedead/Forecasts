# Live-Simulated Trading Agent

This repository provides a scaffold for experimenting with intraday momentum trading strategies
in a safe, simulated environment.  The framework focuses on modular components so that different
strategies, data sources, and analytics can be developed independently and then plugged together.

## Features

- **Rule-based strategy** implementing classic gap-and-go and pullback style heuristics.
- **Reinforcement-learning strategy** that wraps a Stable Baselines3 policy.
- **Hybrid strategy** combining deterministic filters with RL confirmation.
- **Simulation engine** that executes trades with position sizing and 2:1 reward-to-risk targeting.
- **Backtesting helper** for running strategies over historical CSV data.
- **Metrics module** that records trades and calculates performance statistics such as win rate,
  profit factor, and drawdowns.
- **Command-line interface** to switch between strategies and modes quickly.

## Repository Structure

```
Forecasts/
├── backtesting/
│   └── backtester.py
├── data/
│   ├── data_feed.py
│   └── features.py
├── metrics/
│   └── metrics.py
├── simulation/
│   └── engine.py
├── strategies/
│   ├── hybrid.py
│   ├── rl_agent.py
│   └── rule_based.py
├── main.py
└── README.md
```

Each folder contains heavily commented scaffolding so that you can flesh out production-quality
implementations as you experiment with new ideas.

## Getting Started

1. **Install dependencies**

   Create a virtual environment and install packages required by the scaffold.  At minimum you will
   need `pandas`, `numpy`, and optionally `yfinance` and `stable-baselines3` if you plan to fetch
   live data or use the RL agent.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install pandas numpy yfinance stable-baselines3
   ```

2. **Download or generate data**

   Backtests expect a CSV file with the columns `datetime,open,high,low,close,volume`.
   The `data/data_feed.py` module contains helpers for downloading from Yahoo Finance or the
   Alpha Vantage API.

3. **Run a backtest**

   ```bash
   python main.py --strategy rule_based --mode backtest --symbol AAPL --datafile data/aapl_sample.csv
   ```

   The script prints a summary containing trade count, win rate, max drawdown, and other metrics.

4. **Simulate live**

   ```bash
   python main.py --strategy hybrid --mode live --symbol TSLA
   ```

   Live mode streams data through the `DataStreamer` class.  The current implementation polls
   Yahoo Finance and is intended for experimentation rather than production use.

## Notes

- The scaffold deliberately omits advanced error handling and production concerns.  Add proper
  logging, configuration management, and API key storage as you expand the project.
- When training RL agents you will need to supply a `models/trained_agent.zip` file; the repository
  does not include a pre-trained model.
- For serious quantitative research consider integrating proper unit tests and CI pipelines.

## License

This scaffold is released for educational purposes.  Use it to prototype strategies and learn,
but never risk capital without thorough testing and regulatory compliance.
