"""Command line interface for the live trading simulator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from backtesting.backtester import Backtester
from simulation.engine import SimulationEngine
from strategies.hybrid import HybridStrategy
from strategies.rl_agent import RLAgentStrategy
from strategies.rule_based import RuleBasedStrategy


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Trading Agent Simulator")
    parser.add_argument("--strategy", choices=["rule_based", "rl_agent", "hybrid"], default="rule_based")
    parser.add_argument("--mode", choices=["live", "backtest"], default="backtest")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Ticker symbol to trade")
    parser.add_argument("--datafile", type=Path, help="CSV file with historical data for backtests")
    parser.add_argument("--model", type=Path, default=Path("models/trained_agent.zip"), help="Path to a Stable Baselines3 model")
    return parser.parse_args(argv)


def build_strategy(name: str, model_path: Path) -> object:
    if name == "rule_based":
        return RuleBasedStrategy()
    if name == "rl_agent":
        return RLAgentStrategy(model_path=str(model_path))
    if name == "hybrid":
        rule = RuleBasedStrategy()
        rl = RLAgentStrategy(model_path=str(model_path))
        return HybridStrategy(rule_strategy=rule, rl_strategy=rl)
    raise ValueError(f"Unsupported strategy: {name}")


def run_backtest(strategy: object, datafile: Path) -> None:
    if not datafile or not datafile.exists():
        print("Please provide a valid --datafile for backtesting.")
        return

    dataframe = pd.read_csv(datafile, parse_dates=["datetime"])
    backtester = Backtester(strategy=strategy)
    print(f"Running backtest on {datafile} using {strategy.__class__.__name__}...")
    results = backtester.run_backtest(dataframe)
    print("Backtest results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


def run_live(strategy: object, symbol: str) -> None:
    engine = SimulationEngine(strategy=strategy)
    print(f"Starting live simulation for {symbol} using {strategy.__class__.__name__}...")
    try:
        summary = engine.run(symbol, live=True)
    except KeyboardInterrupt:
        print("\nStopping live simulation...")
        summary = engine.run(symbol, live=False)
    print("Final metrics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    strategy = build_strategy(args.strategy, args.model)
    if args.mode == "backtest":
        run_backtest(strategy, args.datafile)
    else:
        run_live(strategy, args.symbol)


if __name__ == "__main__":
    main()
