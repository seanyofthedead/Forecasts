"""Tests for the reinforcement-learning strategy wrapper."""
from __future__ import annotations

import numpy as np

from strategies.rl_agent import RLAgentStrategy
from tests.conftest import DummyModel


def test_decide_returns_mapping_from_model_outputs():
    strategy = RLAgentStrategy()
    strategy.model = DummyModel([1, 2, 0])  # buy, sell, hold

    tick_state = {"prices": [1.0, 2.0, 3.0], "position": 0}
    assert strategy.decide(tick_state) == ("BUY", "rl_signal")
    assert strategy.decide(tick_state) == ("SELL", "rl_signal")
    assert strategy.decide(tick_state) is None


def test_decide_returns_none_without_model():
    strategy = RLAgentStrategy()
    assert strategy.decide({}) is None


def test_state_to_obs_truncates_and_appends_position():
    strategy = RLAgentStrategy()
    prices = list(range(20))
    obs = strategy._state_to_obs({"prices": prices, "position": 1})
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs[-1] == 1
    assert len(obs) == 11  # 10 prices + position


def test_state_to_obs_defaults_to_zero_vector():
    strategy = RLAgentStrategy()
    obs = strategy._state_to_obs({})
    assert obs.tolist() == [0.0]
