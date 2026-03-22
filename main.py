"""Command-line interface for the live simulated trading framework."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backtesting.backtester import Backtester
from strategies.hybrid import HybridStrategy
from strategies.rl_agent import RLAgentStrategy
from strategies.rule_based import RuleBasedStrategy
from simulation.engine import SimulationEngine


def build_strategy(name: str) -> object:
    if name == "rule_based":
        return RuleBasedStrategy()
    if name == "rl_agent":
        return RLAgentStrategy(model_path="models/trained_agent.zip")
    if name == "hybrid":
        return HybridStrategy(RuleBasedStrategy(), RLAgentStrategy(model_path="models/trained_agent.zip"))
    raise ValueError(f"Unsupported strategy '{name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live-Simulated Trading Agent")
    parser.add_argument("--strategy", choices=["rule_based", "rl_agent", "hybrid"], default="rule_based")
    parser.add_argument("--mode", choices=["live", "backtest"], default="backtest")
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--datafile", help="CSV file with historical bars for backtesting")
    args = parser.parse_args()

    strategy = build_strategy(args.strategy)

    if args.mode == "live":
        engine = SimulationEngine(strategy)
        print(f"Starting live simulation on {args.symbol} using {args.strategy} strategy...")
        metrics = engine.run(symbol=args.symbol, live=True)
        print("Final metrics:", metrics)
        return

    if not args.datafile:
        parser.error("--datafile is required in backtest mode")
    data_path = Path(args.datafile)
    if not data_path.exists():
        parser.error(f"Data file {data_path} not found")
    df = pd.read_csv(data_path, parse_dates=["datetime"])
    backtester = Backtester(strategy)
    print(f"Running backtest on {args.symbol} using {args.strategy} strategy")
    results = backtester.run_backtest(df)
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
