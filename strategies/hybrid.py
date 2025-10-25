"""Hybrid strategy combining deterministic rules with an RL agent."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from strategies.rule_based import RuleBasedStrategy
from strategies.rl_agent import RLAgentStrategy

Action = Optional[Tuple[str, str]]


class HybridStrategy:
    """Coordinate rule-based and reinforcement-learning decisions."""

    def __init__(self, rule_strategy: RuleBasedStrategy, rl_strategy: RLAgentStrategy) -> None:
        self.rule_strategy = rule_strategy
        self.rl_strategy = rl_strategy

    def _tick_to_state(self, tick: Dict[str, float]) -> Dict[str, object]:
        prices = tick.get("recent_prices", [tick["close"]])
        position = tick.get("position", 0)
        indicators = [tick.get("vwap", 0.0)]
        return {"prices": prices, "position": position, "indicators": indicators}

    def decide(self, tick: Dict[str, float]) -> Action:
        rule_action = self.rule_strategy.decide(tick)
        rl_action = self.rl_strategy.decide(self._tick_to_state(tick))

        if rule_action and rule_action[0] == "BUY":
            if rl_action and rl_action[0] == "BUY":
                return ("BUY", "hybrid_confirmed_buy")
            return None

        if rule_action and rule_action[0] == "SELL":
            return ("SELL", "hybrid_rule_exit")

        return rl_action
