"""Feature engineering helpers used by trading strategies."""
from __future__ import annotations

from typing import Iterable

import pandas as pd


def calc_relative_volume(current_volume: float, avg_volume: float) -> float:
    """Return the relative volume ratio (current divided by average)."""
    if avg_volume <= 0:
        return float("inf")
    return current_volume / avg_volume


def calc_gap_percent(current_price: float, previous_close: float) -> float:
    """Return the percentage gap between ``current_price`` and ``previous_close``."""
    if previous_close == 0:
        return 0.0
    return (current_price - previous_close) / previous_close * 100.0


def calc_moving_average(prices: Iterable[float], window: int = 20) -> float:
    """Compute a simple moving average over the last ``window`` prices."""
    series = pd.Series(list(prices))
    if len(series) < window:
        return float(series.mean())
    return float(series.tail(window).mean())
