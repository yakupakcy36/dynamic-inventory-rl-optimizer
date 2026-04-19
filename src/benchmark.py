"""Comparative benchmark: RL Agent vs Fixed (s, S) Heuristic vs Naive JIT."""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from src.config import InventoryRLConfig, DEFAULT_RL_CONFIG
from src.environment import InventoryGymEnv
from src.agent import QLearningInventoryAgent


@dataclass
class PolicyEvaluationResult:
    policy_name: str
    total_cost_usd: float
    holding_cost_usd: float
    stockout_penalty_usd: float
    order_cost_usd: float
    service_level_fill_rate_pct: float
    stockout_days: int


def evaluate_policy(
    policy_type: str,
    agent: QLearningInventoryAgent = None,
    eval_days: int = 180,
    seed: int = 999,
    config: InventoryRLConfig = DEFAULT_RL_CONFIG,
    s_reorder: int = 25,
    S_order_up_to: int = 60,
) -> PolicyEvaluationResult:
    """Evaluates a chosen inventory replenishment policy across fixed stochastic demand horizon."""
    env = InventoryGymEnv(config=config, seed=seed)
    env.config = InventoryRLConfig(**{**config.__dict__, "episode_length_days": eval_days})
    state = env.reset()

    total_holding = 0.0
    total_stockout = 0.0
    total_ordering = 0.0
    stockout_count = 0

    for day in range(eval_days):
        on_hand = env.on_hand
        in_transit = sum(qty for arr_day, qty in env.pipeline_orders if arr_day > env.day)
        inventory_position = on_hand + in_transit

        if policy_type == "RL_AGENT":
            action = agent.select_action(state, greedy=True)
        elif policy_type == "FIXED_s_S":
            # Traditional (s, S) policy: if inventory position <= s, order up to S
            if inventory_position <= s_reorder:
                order_qty = max(0, S_order_up_to - inventory_position)
                # Map to closest discrete step
                action = min(env.action_space, key=lambda a: abs(a - order_qty))
            else:
                action = 0
        elif policy_type == "NAIVE_JIT":
            # Order yesterday's demand
            yesterday_d = env.recent_demand
            action = min(env.action_space, key=lambda a: abs(a - yesterday_d))
        else:
            raise ValueError(f"Unknown policy: {policy_type}")

        res = env.step(action)
        state = res.next_state

        # Decompose costs based on step dynamics
        o_cost = (config.fixed_order_cost if action > 0 else 0.0) + (config.unit_purchase_cost * action)
        h_cost = config.holding_cost_per_unit_day * env.on_hand
        p_cost = max(0.0, res.info["day_cost"] - (h_cost + o_cost))

        total_holding += h_cost
        total_stockout += p_cost
        total_ordering += o_cost
        if res.info["stockout"]:
            stockout_count += 1

    total_cost = total_holding + total_stockout + total_ordering
    fill_rate = (env.total_demand_satisfied / max(env.total_demand_seen, 1)) * 100.0

    return PolicyEvaluationResult(
        policy_name=policy_type,
        total_cost_usd=round(total_cost, 2),
        holding_cost_usd=round(total_holding, 2),
        stockout_penalty_usd=round(total_stockout, 2),
        order_cost_usd=round(total_ordering, 2),
        service_level_fill_rate_pct=round(fill_rate, 1),
        stockout_days=stockout_count,
    )
