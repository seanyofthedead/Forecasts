"""Tests for the deterministic rule-based strategy."""
from __future__ import annotations

from datetime import datetime

from strategies.rule_based import RuleBasedStrategy


def _base_tick(**overrides):
    tick = {
        "datetime": datetime(2024, 1, 2, 9, 30),
        "close": 10.0,
        "volume": 10_000,
        "avg_volume": 1_000,
        "float": 1_000_000,
    }
    tick.update(overrides)
    return tick


def test_rejects_ticks_that_fail_filters():
    strategy = RuleBasedStrategy()
    tick = _base_tick(close=50.0)  # above max_stock_price
    assert strategy.decide(tick) is None

    tick = _base_tick(close=10.0, volume=1_000, avg_volume=10_000)  # low relative volume
    assert strategy.decide(tick) is None


def test_gap_breakout_signal_triggers_buy():
    strategy = RuleBasedStrategy(premarket_high=9.5, previous_close=8.0)
    tick = _base_tick(close=10.0)
    action = strategy.decide(tick)
    assert action == ("BUY", "gap_breakout")


def test_pullback_signal_after_recent_prices():
    strategy = RuleBasedStrategy(min_rel_volume=1.0)
    strategy.premarket_high = 100.0  # ensure breakout path not taken
    # Feed enough ticks to build up moving average window
    prices = [10.0, 10.5, 10.2, 10.4, 10.3]
    action = None
    for price in prices:
        action = strategy.decide(_base_tick(close=price, volume=10_000))
    assert action == ("BUY", "pullback_continuation")
