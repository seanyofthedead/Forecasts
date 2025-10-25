"""Reinforcement learning driven trading strategy."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Optional import (Stable Baselines3 is not installed by default)
try:  # pragma: no cover - optional dependency
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover
    PPO = None

Action = Optional[Tuple[str, str]]


class RLAgentStrategy:
    """Thin wrapper around a pre-trained Stable Baselines3 policy."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        if self.model_path and PPO is not None and self.model_path.exists():
            try:
                self.model = PPO.load(self.model_path)
            except Exception as exc:  # pragma: no cover - depends on external files
                print(f"Failed to load RL model from {self.model_path}: {exc}")

    def decide(self, state: Dict) -> Action:
        if self.model is None:
            return None

        observation = self._state_to_obs(state)
        action_idx, _ = self.model.predict(observation, deterministic=True)
        mapping = {0: None, 1: ("BUY", "rl_signal"), 2: ("SELL", "rl_signal")}
        return mapping.get(int(action_idx))

    def _state_to_obs(self, state: Dict) -> np.ndarray:
        prices = state.get("prices", [])
        position = state.get("position", 0)
        features = list(prices[-10:]) + [position]
        if not features:
            features = [0.0]
        return np.asarray(features, dtype=np.float32)
