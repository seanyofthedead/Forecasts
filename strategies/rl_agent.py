"""Reinforcement-learning strategy wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:  # Optional dependency
    from stable_baselines3 import PPO  # type: ignore
except Exception:  # pragma: no cover
    PPO = None

Action = Optional[Tuple[str, str]]


class RLAgentStrategy:
    """Use a Stable Baselines3 policy to determine trade actions."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        if self.model_path and self.model_path.exists() and PPO:
            try:
                self.model = PPO.load(str(self.model_path))
            except Exception as exc:  # pragma: no cover - load failures are user specific
                print(f"Failed to load RL model at {self.model_path}: {exc}")

    def decide(self, state: Dict[str, object]) -> Action:
        if not self.model:
            return None
        observation = self._state_to_observation(state)
        action_idx, _ = self.model.predict(observation, deterministic=True)
        if int(action_idx) == 1:
            return ("BUY", "rl_agent_buy")
        if int(action_idx) == 2:
            return ("SELL", "rl_agent_sell")
        return None

    def _state_to_observation(self, state: Dict[str, object]) -> np.ndarray:
        prices = state.get("prices", [])
        position = state.get("position", 0)
        indicators = state.get("indicators", [])
        vector = np.array(list(prices)[-10:] + [position] + list(indicators), dtype=np.float32)
        if vector.size == 0:
            vector = np.zeros(1, dtype=np.float32)
        return vector
