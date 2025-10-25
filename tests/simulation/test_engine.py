"""Tests for the simulation engine trade loop."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from simulation import engine as engine_module
from simulation.engine import SimulationEngine
from tests.conftest import SequenceStrategy


class DummyStreamer:
    def __init__(self, ticks):
        self._ticks = list(ticks)

    def stream(self, live: bool = False):  # pragma: no cover - trivial generator
        for tick in self._ticks:
            yield tick


def _ticks_with_prices(prices):
    base = datetime(2024, 1, 2, 9, 30)
    for offset, price in enumerate(prices):
        yield {
            "datetime": base + timedelta(minutes=offset),
            "close": price,
            "volume": 10_000,
            "avg_volume": 1_000,
        }


def test_engine_executes_buy_and_targets_profit(monkeypatch):
    ticks = list(_ticks_with_prices([100.0, 105.0]))
    monkeypatch.setattr(engine_module, "DataStreamer", lambda *a, **k: DummyStreamer(ticks))
    strategy = SequenceStrategy(actions=[("BUY", "test"), None])
    engine = SimulationEngine(strategy=strategy)

    summary = engine.run("TEST")
    assert engine.trade_log[0]["pnl"] == pytest.approx(250.0)
    assert summary["Net Profit"] == 250.0


def test_engine_hits_stop_loss(monkeypatch):
    ticks = list(_ticks_with_prices([100.0, 95.0]))
    monkeypatch.setattr(engine_module, "DataStreamer", lambda *a, **k: DummyStreamer(ticks))
    strategy = SequenceStrategy(actions=[("BUY", "test"), None])
    engine = SimulationEngine(strategy=strategy)

    summary = engine.run("TEST")
    assert engine.trade_log[0]["pnl"] == pytest.approx(-250.0)
    assert summary["Net Profit"] == -250.0


def test_engine_ignores_additional_buys_while_in_position(monkeypatch):
    ticks = list(_ticks_with_prices([100.0, 101.0, 102.0]))
    monkeypatch.setattr(engine_module, "DataStreamer", lambda *a, **k: DummyStreamer(ticks))
    strategy = SequenceStrategy(actions=[("BUY", "test"), ("BUY", "test"), ("BUY", "test")])
    engine = SimulationEngine(strategy=strategy)

    summary = engine.run("TEST")
    assert len(engine.trade_log) == 1
    assert summary["Total Trades"] == 1
