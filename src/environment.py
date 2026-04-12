"""Gym-style Markov Decision Process (MDP) for Dynamic Inventory Optimization."""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, List
import numpy as np
from src.config import InventoryRLConfig, DEFAULT_RL_CONFIG


@dataclass
class StepResult:
    next_state: Tuple[int, int, int]
    reward: float
    done: bool
    info: Dict[str, Any]


class InventoryGymEnv:
    """Stochastic lead-time and Poisson demand inventory environment."""

    def __init__(self, config: InventoryRLConfig = DEFAULT_RL_CONFIG, seed: int = 42):
        self.config = config
        self.np_random = np.random.RandomState(seed)
        self.action_space = [0, 5, 10, 15, 20, 25, 30, 40]
        self.reset()

    def reset(self) -> Tuple[int, int, int]:
        """Resets warehouse state to initial conditions."""
        self.day = 0
        self.on_hand = self.config.initial_inventory
        self.pipeline_orders: List[Tuple[int, int]] = []  # List of (arrival_day, quantity)
        self.recent_demand = int(self.config.demand_mean_poisson)
        self.cumulative_cost = 0.0
        self.stockout_events = 0
        self.total_demand_seen = 0
        self.total_demand_satisfied = 0
        return self._get_state()

    def _discretize(self, on_hand: int, in_transit: int, demand: int) -> Tuple[int, int, int]:
        """Discretizes continuous states into compact finite state buckets."""
        # On-hand bucket: clamped between -30 and 150, step of 10
        oh_clamped = int(np.clip(on_hand, -30, self.config.max_warehouse_capacity))
        oh_bucket = int(oh_clamped // 10)

        # In-transit bucket: clamped 0 to 80, step of 10
        it_clamped = int(np.clip(in_transit, 0, 80))
        it_bucket = int(it_clamped // 10)

        # Demand bucket: step of 5
        d_clamped = int(np.clip(demand, 0, 30))
        d_bucket = int(d_clamped // 5)

        return (oh_bucket, it_bucket, d_bucket)

    def _get_state(self) -> Tuple[int, int, int]:
        in_transit = sum(qty for arr_day, qty in self.pipeline_orders if arr_day > self.day)
        return self._discretize(self.on_hand, in_transit, self.recent_demand)

    def step(self, action: int) -> StepResult:
        """Executes one day of supply chain dynamics."""
        self.day += 1

        # 1. Receive incoming shipments scheduled to arrive on or before today
        arrived_qty = sum(qty for arr_day, qty in self.pipeline_orders if arr_day == self.day)
        self.pipeline_orders = [(arr_day, qty) for arr_day, qty in self.pipeline_orders if arr_day > self.day]
        self.on_hand += arrived_qty

        # 2. Place new order with stochastic lead time
        if action > 0:
            lead_time = int(self.np_random.randint(self.config.min_lead_time_days, self.config.max_lead_time_days + 1))
            self.pipeline_orders.append((self.day + lead_time, action))

        # 3. Realize customer demand from Poisson distribution
        realized_demand = int(self.np_random.poisson(self.config.demand_mean_poisson))
        self.recent_demand = realized_demand
        self.total_demand_seen += realized_demand

        # Satisfy demand
        if self.on_hand >= realized_demand:
            self.total_demand_satisfied += realized_demand
        else:
            self.total_demand_satisfied += max(0, self.on_hand)
            self.stockout_events += 1

        self.on_hand -= realized_demand

        # 4. Asymmetric Economic Cost Function
        holding_cost = self.config.holding_cost_per_unit_day * max(0, self.on_hand)
        stockout_penalty = self.config.stockout_penalty_per_unit * max(0, -self.on_hand)
        order_cost = (self.config.fixed_order_cost if action > 0 else 0.0) + (self.config.unit_purchase_cost * action)

        day_cost = holding_cost + stockout_penalty + order_cost
        self.cumulative_cost += day_cost

        # Reward is negative cost (agent maximizes reward = minimizes total supply chain cost)
        reward = -float(day_cost)
        done = (self.day >= self.config.episode_length_days)

        fill_rate = (self.total_demand_satisfied / max(self.total_demand_seen, 1)) * 100.0

        info = {
            "day": self.day,
            "on_hand": self.on_hand,
            "demand": realized_demand,
            "action": action,
            "day_cost": round(day_cost, 2),
            "fill_rate_pct": round(fill_rate, 1),
            "stockout": self.on_hand < 0,
        }

        return StepResult(next_state=self._get_state(), reward=reward, done=done, info=info)
