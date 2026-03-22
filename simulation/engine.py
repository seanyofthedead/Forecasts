"""Simulation engine that executes strategies against a data stream."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from data.data_feed import DataStreamer
from metrics import metrics

@dataclass
class SimulationEngine:
    """Simple long-only execution engine."""

    strategy: object
    initial_cash: float = 10_000.0
    risk_per_trade: float = 0.01
    trade_log: List[Dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_cash
        self.position = 0
        self.entry_price: Optional[float] = None
        self.stop_price: Optional[float] = None
        self.target_price: Optional[float] = None

    def run(self, symbol: str, live: bool = False, historical_path: Optional[str] = None) -> Dict[str, float]:
        streamer = DataStreamer(symbol=symbol, historical_path=historical_path)
        last_price: Optional[float] = None
        last_timestamp = None
        for tick in streamer.stream(continuous=live):
            tick_dict = tick.as_dict()
            tick_dict["position"] = self.position

            last_price = tick_dict["close"]
            last_timestamp = tick_dict["datetime"]
            self._check_open_position(tick_dict)

            action = self.strategy.decide(tick_dict)
            if action:
                verb, reason = action
                if verb == "BUY" and self.position == 0:
                    shares = self._compute_position_size(last_price)
                    if shares > 0:
                        self._open_position(shares, last_price, tick_dict["datetime"], reason)
                elif verb == "SELL" and self.position > 0:
                    self._close_position(last_price, tick_dict["datetime"], reason)
        if self.position and last_price is not None and last_timestamp is not None:
            self._close_position(last_price, last_timestamp, "end_of_stream")
        return metrics.compute_metrics(self.trade_log)

    def _compute_position_size(self, price: float) -> int:
        risk_amount = self.cash * self.risk_per_trade
        stop_distance = max(price * 0.02, 0.01)
        shares = int(risk_amount / stop_distance)
        if shares < 1:
            return 0
        self.stop_price = price - stop_distance
        self.target_price = price + (2 * stop_distance)
        return shares

    def _open_position(self, shares: int, price: float, timestamp, reason: str) -> None:
        cost = shares * price
        if cost > self.cash:
            shares = int(self.cash // price)
            cost = shares * price
        if shares < 1:
            return
        self.cash -= cost
        self.position = shares
        self.entry_price = price
        trade = {
            "entry_time": timestamp,
            "exit_time": None,
            "action": "BUY",
            "shares": shares,
            "entry_price": price,
            "exit_price": None,
            "pnl": None,
            "reason": reason,
        }
        self.trade_log.append(trade)

    def _close_position(self, price: float, timestamp, reason: str) -> None:
        if self.position == 0 or self.entry_price is None:
            return
        shares = self.position
        pnl = shares * (price - self.entry_price)
        self.cash += shares * price
        trade = self.trade_log[-1]
        trade.update({"exit_time": timestamp, "exit_price": price, "pnl": pnl, "reason": f"{trade['reason']}|exit:{reason}"})
        self.position = 0
        self.entry_price = None
        self.stop_price = None
        self.target_price = None

    def _check_open_position(self, tick: Dict[str, float]) -> None:
        if self.position == 0 or self.entry_price is None:
            return
        price = tick["close"]
        if self.stop_price and price <= self.stop_price:
            self._close_position(price, tick["datetime"], "stop_loss")
        elif self.target_price and price >= self.target_price:
            self._close_position(price, tick["datetime"], "take_profit")
