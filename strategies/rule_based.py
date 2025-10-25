"""Deterministic momentum trading strategy implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

from data import features

Action = Optional[Tuple[str, str]]


@dataclass
class MarketContext:
    """Holds pre-market information used by the strategy."""

    previous_close: Optional[float] = None
    premarket_high: Optional[float] = None
    average_volume: Optional[float] = None
    float_shares: Optional[float] = None


@dataclass
class RuleBasedStrategy:
    """Implements simplified gap-and-go and pullback style rules."""

    max_stock_price: float = 20.0
    min_stock_price: float = 1.0
    min_rel_volume: float = 5.0
    min_gap_percent: float = 5.0
    max_float: float = 50_000_000
    context: MarketContext = field(default_factory=MarketContext)
    last_pullback_low: Optional[float] = None
    last_decision_time: Optional[datetime] = None

    def prepare_for_symbol(self, metadata: Dict[str, float]) -> None:
        """Initialize context from metadata such as previous close and float."""
        self.context.previous_close = metadata.get("prev_close")
        self.context.premarket_high = metadata.get("premarket_high")
        self.context.average_volume = metadata.get("avg_volume")
        self.context.float_shares = metadata.get("float")

    def _passes_filters(self, price: float, volume: float, avg_volume: Optional[float]) -> bool:
        if price < self.min_stock_price or price > self.max_stock_price:
            return False
        if self.context.float_shares and self.context.float_shares > self.max_float:
            return False
        if avg_volume:
            rel_volume = features.calc_relative_volume(volume, avg_volume)
            if rel_volume < self.min_rel_volume:
                return False
        return True

    def _is_opening_tick(self, timestamp: datetime) -> bool:
        return timestamp.strftime("%H:%M") == "09:30"

    def decide(self, tick: Dict[str, float]) -> Action:
        """Return an action tuple ``("BUY"|"SELL", reason)`` or ``None``."""
        price = tick["close"]
        volume = tick["volume"]
        timestamp = tick["datetime"]
        avg_volume = tick.get("avg_volume", self.context.average_volume)

        if not self._passes_filters(price, volume, avg_volume):
            return None

        if self.context.previous_close and self._is_opening_tick(timestamp):
            gap_pct = features.calc_gap_percent(price, self.context.previous_close)
            if gap_pct < self.min_gap_percent:
                return None

        if self.context.premarket_high and price > self.context.premarket_high and self._is_opening_tick(timestamp):
            self.last_decision_time = timestamp
            return ("BUY", "gap_breakout")

        pullback_trigger = False
        if self.last_pullback_low is not None and price > self.last_pullback_low * 1.01:
            pullback_trigger = True
        if tick.get("intraday_low") is not None:
            intraday_low = tick["intraday_low"]
            if self.last_pullback_low is None or intraday_low < self.last_pullback_low:
                self.last_pullback_low = intraday_low

        if pullback_trigger:
            self.last_decision_time = timestamp
            return ("BUY", "pullback_continuation")

        if tick.get("position", 0) > 0 and tick.get("vwap") and price < tick["vwap"] * 0.99:
            return ("SELL", "lost_vwap")

        return None
