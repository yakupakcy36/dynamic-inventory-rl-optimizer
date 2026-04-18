"""Unit tests for comparative policy evaluation."""

from src.agent import QLearningInventoryAgent
from src.benchmark import evaluate_policy


def test_evaluate_policies():
    agent = QLearningInventoryAgent()
    agent.train(episodes=50)

    for p in ["RL_AGENT", "FIXED_s_S", "NAIVE_JIT"]:
        res = evaluate_policy(p, agent=agent, eval_days=30, seed=42)
        assert res.total_cost_usd > 0
        assert 0.0 <= res.service_level_fill_rate_pct <= 100.0
        assert res.stockout_days >= 0
