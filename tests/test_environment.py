"""Unit tests for Inventory Gym environment."""

from src.config import DEFAULT_RL_CONFIG
from src.environment import InventoryGymEnv


def test_env_reset():
    env = InventoryGymEnv(config=DEFAULT_RL_CONFIG, seed=42)
    state = env.reset()
    assert isinstance(state, tuple)
    assert len(state) == 3
    assert env.on_hand == DEFAULT_RL_CONFIG.initial_inventory
    assert env.day == 0


def test_env_step_mechanics():
    env = InventoryGymEnv(config=DEFAULT_RL_CONFIG, seed=42)
    state = env.reset()
    action = 10
    res = env.step(action)

    assert isinstance(res.next_state, tuple)
    assert res.reward < 0.0  # Reward is negative cost
    assert isinstance(res.done, bool)
    assert "fill_rate_pct" in res.info
    assert res.info["day"] == 1
