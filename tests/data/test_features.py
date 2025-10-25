"""Tests for feature engineering helpers."""
from __future__ import annotations

import math

import pandas as pd

from data import features


def test_calc_relative_volume_handles_zero_average():
    assert features.calc_relative_volume(1_000, 0) == math.inf
    assert math.isclose(features.calc_relative_volume(2_000, 1_000), 2.0)


def test_calc_gap_percent_zero_previous_close():
    assert features.calc_gap_percent(100, 0) == 0.0
    assert math.isclose(features.calc_gap_percent(110, 100), 10.0)
    assert math.isclose(features.calc_gap_percent(90, 100), -10.0)


def test_calc_moving_average_windowing():
    prices = [1, 2, 3, 4, 5]
    assert math.isclose(features.calc_moving_average(prices, window=10), pd.Series(prices).mean())
    assert math.isclose(features.calc_moving_average(prices, window=3), pd.Series(prices[-3:]).mean())
