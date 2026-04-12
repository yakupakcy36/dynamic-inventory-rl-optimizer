"""Configuration parameters for Dynamic Inventory Reinforcement Learning Optimizer."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InventoryRLConfig:
    # Warehouse Economic Parameters ($ per unit)
    holding_cost_per_unit_day: float = 2.00
    stockout_penalty_per_unit: float = 45.00
    fixed_order_cost: float = 25.00
    unit_purchase_cost: float = 10.00

    # Physical Warehouse & Supplier Constraints
    max_warehouse_capacity: int = 150
    initial_inventory: int = 40
    demand_mean_poisson: float = 12.0
    min_lead_time_days: int = 1
    max_lead_time_days: int = 3

    # Q-Learning Hyperparameters
    alpha_learning_rate: float = 0.15
    gamma_discount_factor: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.9992
    training_episodes: int = 2500
    episode_length_days: int = 60

    # Storage paths
    artifacts_dir: Path = Path("artifacts")
    model_filename: str = "inventory_q_policy.joblib"


DEFAULT_RL_CONFIG = InventoryRLConfig()
