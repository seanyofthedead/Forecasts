"""Rule-based day trading strategy implementations."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

from data import features

Action = Optional[Tuple[str, str]]


@dataclass
class RuleBasedStrategy:
    """Deterministic momentum strategy focusing on gap-and-go and pullbacks."""

    max_stock_price: float = 20.0
    min_stock_price: float = 1.0
    min_rel_volume: float = 5.0
    min_gap_percent: float = 5.0
    max_float: float = 50_000_000

    premarket_high: Optional[float] = None
    previous_close: Optional[float] = None
    average_volume: Optional[float] = None
    recent_prices: list = field(default_factory=list)

    def prepare_for_symbol(self, metadata: Dict[str, float]) -> None:
        """Provide pre-market context before intraday ticks arrive."""
        self.premarket_high = metadata.get("premarket_high")
        self.previous_close = metadata.get("prev_close")
        self.average_volume = metadata.get("avg_volume")

    def _passes_filters(self, price: float, tick: Dict) -> bool:
        if price < self.min_stock_price or price > self.max_stock_price:
            return False

        if tick.get("float") and tick["float"] > self.max_float:
            return False

        avg_vol = tick.get("avg_volume", self.average_volume)
        if avg_vol is not None:
            rel_volume = features.calc_relative_volume(tick.get("volume", 0), avg_vol)
            if rel_volume < self.min_rel_volume:
                return False

        timestamp = tick.get("datetime")
        if isinstance(timestamp, datetime) and timestamp.hour == 9 and timestamp.minute == 30:
            if self.previous_close is not None:
                gap = features.calc_gap_percent(price, self.previous_close)
                if gap < self.min_gap_percent:
                    return False
        return True

    def _detect_breakout(self, price: float, tick: Dict) -> Action:
        timestamp = tick.get("datetime")
        if (
            self.premarket_high
            and price > self.premarket_high
            and isinstance(timestamp, datetime)
            and timestamp.hour == 9
            and timestamp.minute == 30
        ):
            return ("BUY", "gap_breakout")
        return None

    def _detect_pullback(self, price: float) -> Action:
        self.recent_prices.append(price)
        window = min(len(self.recent_prices), 10)
        if window < 5:
            return None
        moving_average = features.calc_moving_average(self.recent_prices[-window:], window=window)
        if price >= moving_average and price <= max(self.recent_prices[-window:]):
            return ("BUY", "pullback_continuation")
        return None

    def decide(self, tick: Dict) -> Action:
        """Return the next action based on the incoming tick."""
        price = float(tick.get("close", 0.0))
        if not self._passes_filters(price, tick):
            return None

        breakout = self._detect_breakout(price, tick)
        if breakout:
            return breakout

        return self._detect_pullback(price)
