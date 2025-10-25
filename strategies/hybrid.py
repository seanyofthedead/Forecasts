"""Hybrid strategy combining deterministic rules with an RL policy."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from strategies.rl_agent import RLAgentStrategy
from strategies.rule_based import RuleBasedStrategy

Action = Optional[Tuple[str, str]]


class HybridStrategy:
    """Coordinate rule-based and RL strategies to form a combined signal."""

    def __init__(self, rule_strategy: RuleBasedStrategy, rl_strategy: RLAgentStrategy):
        self.rule_strategy = rule_strategy
        self.rl_strategy = rl_strategy

    def _tick_to_state(self, tick: Dict, position: int = 0) -> Dict:
        return {"prices": [tick.get("close", 0.0)], "position": position}

    def decide(self, tick: Dict, position: int = 0) -> Action:
        rule_action = self.rule_strategy.decide(tick)
        rl_action = self.rl_strategy.decide(self._tick_to_state(tick, position))

        if rule_action and rule_action[0] == "BUY":
            if rl_action and rl_action[0] == "BUY":
                return ("BUY", "hybrid_confirmed")
            return None

        if rule_action and rule_action[0] == "SELL":
            return ("SELL", "hybrid_rule_exit")

        return rl_action
