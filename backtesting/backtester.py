"""Backtesting harness for the trading simulator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from metrics import metrics
from simulation.engine import SimulationEngine


@dataclass
class Backtester:
    strategy: object

    def run_backtest(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        engine = SimulationEngine(strategy=self.strategy, initial_cash=10_000.0)
        latest_row: Optional[pd.Series] = None
        for _, row in historical_data.iterrows():
            latest_row = row
            tick = {
                "datetime": row.get("datetime"),
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume"),
            }

            price = float(tick["close"])

            if engine.position and engine.position_entry_price is not None:
                if engine.stop_price is not None and price <= engine.stop_price:
                    engine._close_position(price, tick.get("datetime"), reason="stop_loss")
                elif engine.target_price is not None and price >= engine.target_price:
                    engine._close_position(price, tick.get("datetime"), reason="take_profit")

            try:
                action = self.strategy.decide(tick)  # type: ignore[arg-type]
            except TypeError:
                action = self.strategy.decide(tick, engine.position)

            if action:
                side, reason = action
                if side == "BUY" and engine.position == 0:
                    shares = engine._compute_position_size(price)
                    if shares:
                        engine._open_position(shares, price, tick.get("datetime"), reason=reason)
                elif side == "SELL" and engine.position > 0:
                    engine._close_position(price, tick.get("datetime"), reason=reason)

        if latest_row is not None and engine.position:
            engine._close_position(float(latest_row.get("close", 0.0)), latest_row.get("datetime"), reason="end_of_backtest")

        return metrics.compute_metrics(engine.trade_log)
