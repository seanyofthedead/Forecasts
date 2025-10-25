"""Simulation engine that wires together data feeds and trading strategies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from data.data_feed import DataStreamer
from metrics import metrics


@dataclass
class Trade:
    action: str
    shares: int
    entry_price: float
    entry_time: Optional[object]
    exit_price: Optional[float] = None
    exit_time: Optional[object] = None
    pnl: Optional[float] = None
    reason: str = ""


@dataclass
class SimulationEngine:
    """Drive a strategy using streaming market data."""

    strategy: object
    initial_cash: float = 10_000.0
    risk_per_trade: float = 0.01
    poll_interval: float = 1.0

    cash: float = field(init=False)
    position: int = field(init=False, default=0)
    position_entry_price: Optional[float] = field(init=False, default=None)
    stop_price: Optional[float] = field(init=False, default=None)
    target_price: Optional[float] = field(init=False, default=None)
    trade_log: List[Dict] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_cash

    def run(self, symbol: str, *, live: bool = False) -> Dict:
        streamer = DataStreamer(symbol, source="yahoo", poll_interval=self.poll_interval)
        latest_tick: Optional[Dict] = None
        for tick in streamer.stream(live=live):
            latest_tick = tick
            price = float(tick.get("close", 0.0))

            if self.position and self.position_entry_price is not None:
                if self.stop_price is not None and price <= self.stop_price:
                    self._close_position(price, tick.get("datetime"), reason="stop_loss")
                elif self.target_price is not None and price >= self.target_price:
                    self._close_position(price, tick.get("datetime"), reason="take_profit")

            action = None
            if hasattr(self.strategy, "decide"):
                if isinstance(tick, dict):
                    try:
                        action = self.strategy.decide(tick)  # type: ignore[arg-type]
                    except TypeError:
                        action = self.strategy.decide(tick, self.position)  # Hybrid signature

            if action:
                side, reason = action
                if side == "BUY" and self.position == 0:
                    shares = self._compute_position_size(price)
                    if shares > 0:
                        self._open_position(shares, price, tick.get("datetime"), reason=reason)
                elif side == "SELL" and self.position > 0:
                    self._close_position(price, tick.get("datetime"), reason=reason)

        if latest_tick and self.position:
            self._close_position(float(latest_tick.get("close", 0.0)), latest_tick.get("datetime"), reason="end_of_stream")

        return metrics.compute_metrics(self.trade_log)

    def _compute_position_size(self, price: float) -> int:
        risk_amount = self.cash * self.risk_per_trade
        stop_loss_price = price * 0.98
        stop_distance = price - stop_loss_price
        if stop_distance <= 0:
            return 0
        shares = max(int(risk_amount // stop_distance), 1)
        self.stop_price = stop_loss_price
        self.target_price = price + 2 * (price - self.stop_price)
        return shares

    def _open_position(self, shares: int, price: float, timestamp, *, reason: str) -> None:
        cost = shares * price
        if cost > self.cash:
            shares = int(self.cash // price)
            cost = shares * price
        if shares <= 0:
            return

        self.cash -= cost
        self.position = shares
        self.position_entry_price = price
        trade = {
            "action": "BUY",
            "shares": shares,
            "entry_price": price,
            "entry_time": timestamp,
            "exit_price": None,
            "exit_time": None,
            "pnl": None,
            "reason": reason,
        }
        self.trade_log.append(trade)
        print(f"[TRADE] Bought {shares} shares at ${price:.2f} ({reason})")

    def _close_position(self, price: float, timestamp, *, reason: str) -> None:
        if self.position == 0:
            return

        shares = self.position
        pnl = shares * (price - (self.position_entry_price or price))
        self.cash += shares * price

        trade = self.trade_log[-1] if self.trade_log else None
        if trade:
            trade["exit_price"] = price
            trade["exit_time"] = timestamp
            trade["pnl"] = pnl
            trade["reason"] = f"{trade['reason']}|exit:{reason}" if trade.get("reason") else reason

        print(f"[TRADE] Sold {shares} shares at ${price:.2f} (exit {reason}), P&L ${pnl:.2f}")

        self.position = 0
        self.position_entry_price = None
        self.stop_price = None
        self.target_price = None
