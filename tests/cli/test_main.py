"""Tests for the command-line interface."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import main as cli


def test_parse_args_populates_defaults():
    args = cli.parse_args([])
    assert args.strategy == "rule_based"
    assert args.mode == "backtest"
    assert args.symbol == "AAPL"


def test_build_strategy_returns_expected_types():
    rule = cli.build_strategy("rule_based", Path("model.zip"))
    rl = cli.build_strategy("rl_agent", Path("model.zip"))
    hybrid = cli.build_strategy("hybrid", Path("model.zip"))
    from strategies.rule_based import RuleBasedStrategy
    from strategies.rl_agent import RLAgentStrategy
    from strategies.hybrid import HybridStrategy

    assert isinstance(rule, RuleBasedStrategy)
    assert isinstance(rl, RLAgentStrategy)
    assert isinstance(hybrid, HybridStrategy)


def test_run_backtest_invokes_backtester(monkeypatch, tmp_path, capsys):
    data = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-02 09:30", periods=2, freq="1min"),
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.0, 101.0],
            "volume": [1_000, 1_100],
        }
    )
    datafile = tmp_path / "data.csv"
    data.to_csv(datafile, index=False)

    captured = {}

    class DummyBacktester:
        def __init__(self, strategy):
            captured["strategy"] = strategy

        def run_backtest(self, dataframe):
            captured["rows"] = len(dataframe)
            return {"Net Profit": 10.0}

    monkeypatch.setattr(cli, "Backtester", DummyBacktester)
    cli.run_backtest(object(), datafile)
    out = capsys.readouterr().out
    assert "Net Profit" in out
    assert captured["rows"] == 2


def test_run_live_invokes_engine(monkeypatch, capsys):
    summary = {"Net Profit": 5.0}

    class DummyEngine:
        def __init__(self, strategy):
            self.strategy = strategy

        def run(self, symbol, live=True):
            assert live is True
            return summary

    monkeypatch.setattr(cli, "SimulationEngine", DummyEngine)
    cli.run_live(object(), "TEST")
    out = capsys.readouterr().out
    assert "Net Profit" in out


def test_main_dispatches_to_modes(monkeypatch):
    called = {"backtest": False, "live": False}

    monkeypatch.setattr(cli, "run_backtest", lambda *a, **k: called.__setitem__("backtest", True))
    monkeypatch.setattr(cli, "run_live", lambda *a, **k: called.__setitem__("live", True))

    cli.main(["--mode", "backtest", "--strategy", "rule_based", "--datafile", "data.csv"])
    assert called["backtest"] is True

    called["backtest"] = False
    cli.main(["--mode", "live", "--strategy", "rule_based"])
    assert called["live"] is True
