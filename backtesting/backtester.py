"""Historical backtesting helper."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from metrics import metrics
from simulation.engine import SimulationEngine


class Backtester:
    """Feed historical bars to a strategy using the :class:`SimulationEngine`."""

    def __init__(self, strategy) -> None:
        self.strategy = strategy

    def run_backtest(self, dataframe: pd.DataFrame) -> Dict[str, float]:
        engine = SimulationEngine(self.strategy, initial_cash=10_000.0)
        for _, row in dataframe.iterrows():
            tick = {
                "datetime": row["datetime"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
                "position": engine.position,
            }
            engine._check_open_position(tick)
            action = self.strategy.decide(tick)
            if action:
                verb, reason = action
                if verb == "BUY" and engine.position == 0:
                    shares = engine._compute_position_size(tick["close"])
                    if shares:
                        engine._open_position(shares, tick["close"], tick["datetime"], reason)
                elif verb == "SELL" and engine.position > 0:
                    engine._close_position(tick["close"], tick["datetime"], reason)
        if engine.position:
            engine._close_position(float(dataframe.iloc[-1]["close"]), dataframe.iloc[-1]["datetime"], "end_of_backtest")
        summary = metrics.compute_metrics(engine.trade_log)
        summary["Final Equity"] = round(engine.cash, 2)
        summary["Starting Equity"] = round(engine.initial_cash, 2)
        return summary
