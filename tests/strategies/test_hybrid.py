"""Tests for the hybrid strategy coordination logic."""
from __future__ import annotations

from strategies.hybrid import HybridStrategy


class StaticRuleStrategy:
    def __init__(self, action):
        self._action = action

    def decide(self, tick):
        return self._action


class StaticRLStrategy:
    def __init__(self, action):
        self._action = action

    def decide(self, state):
        return self._action


def test_hybrid_requires_both_strategies_to_buy():
    hybrid = HybridStrategy(StaticRuleStrategy(("BUY", "rule")), StaticRLStrategy(("BUY", "rl")))
    tick = {"close": 10.0}
    assert hybrid.decide(tick) == ("BUY", "hybrid_confirmed")

    hybrid = HybridStrategy(StaticRuleStrategy(("BUY", "rule")), StaticRLStrategy(None))
    assert hybrid.decide(tick) is None


def test_hybrid_allows_rule_based_exit():
    hybrid = HybridStrategy(StaticRuleStrategy(("SELL", "rule")), StaticRLStrategy(None))
    tick = {"close": 10.0}
    assert hybrid.decide(tick) == ("SELL", "hybrid_rule_exit")


def test_hybrid_falls_back_to_rl_when_rule_is_idle():
    hybrid = HybridStrategy(StaticRuleStrategy(None), StaticRLStrategy(("SELL", "rl")))
    tick = {"close": 10.0}
    assert hybrid.decide(tick) == ("SELL", "rl")
