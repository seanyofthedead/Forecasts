"""Tests for the backtesting harness."""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from backtesting.backtester import Backtester
from tests.conftest import SequenceStrategy


def _historical_frame(prices):
    base = datetime(2024, 1, 2, 9, 30)
    rows = []
    for offset, price in enumerate(prices):
        rows.append(
            {
                "datetime": base + timedelta(minutes=offset),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": 1_000 + offset * 100,
            }
        )
    return pd.DataFrame(rows)


def test_backtester_runs_strategy_over_dataframe():
    dataframe = _historical_frame([100.0, 105.0])
    strategy = SequenceStrategy(actions=[("BUY", "test"), None])
    backtester = Backtester(strategy=strategy)

    results = backtester.run_backtest(dataframe)
    assert results["Total Trades"] == 1
    assert results["Net Profit"] == 250.0


def test_backtester_closes_open_position_at_end():
    dataframe = _historical_frame([100.0, 101.0, 102.0])
    strategy = SequenceStrategy(actions=[("BUY", "test"), None, None])
    backtester = Backtester(strategy=strategy)

    results = backtester.run_backtest(dataframe)
    assert results["Total Trades"] == 1
    # Target was never hit (104), so exit occurs at final close 102 -> profit 100
    assert results["Net Profit"] == 100.0
