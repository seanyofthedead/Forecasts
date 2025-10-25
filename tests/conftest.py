"""Shared fixtures and helpers for trading simulator tests."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional

import pandas as pd
import pytest
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def example_intraday_frame() -> pd.DataFrame:
    """Return a small OHLCV dataframe for use in streaming/backtests."""
    base = datetime(2024, 1, 2, 9, 30)
    rows: List[Dict] = []
    prices = [100.0, 102.0, 104.0, 103.0, 105.0]
    volumes = [1_000, 1_200, 1_500, 1_300, 1_400]
    for index, (price, volume) in enumerate(zip(prices, volumes)):
        rows.append(
            {
                "datetime": base + timedelta(minutes=index),
                "open": price,
                "high": price + 1,
                "low": price - 1,
                "close": price,
                "volume": volume,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class SequenceStrategy:
    """Simple strategy returning a predetermined sequence of actions."""

    actions: Iterable[Optional[tuple]]

    def __post_init__(self) -> None:
        self._actions: Deque[Optional[tuple]] = deque(self.actions)

    def decide(self, tick, position: int | None = None):  # type: ignore[override]
        if not self._actions:
            return None
        return self._actions.popleft()


class DummyModel:
    """Minimal Stable Baselines-like model used for RL strategy tests."""

    def __init__(self, outputs: Iterable[int]):
        self._outputs: Deque[int] = deque(outputs)

    def predict(self, observation, deterministic: bool = True):  # pragma: no cover - simple passthrough
        if self._outputs:
            return self._outputs.popleft(), None
        return 0, None
