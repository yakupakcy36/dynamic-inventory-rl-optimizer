"""Unit tests for Q-Learning agent."""

import tempfile
from pathlib import Path
from src.config import DEFAULT_RL_CONFIG
from src.agent import QLearningInventoryAgent


def test_agent_action_selection_and_update():
    agent = QLearningInventoryAgent(config=DEFAULT_RL_CONFIG)
    state = (3, 1, 2)
    action = agent.select_action(state, greedy=True)
    assert action in agent.actions

    td_err = agent.update(state, action, reward=-25.0, next_state=(2, 1, 2), done=False)
    assert td_err >= 0.0


def test_agent_training_and_serialization():
    agent = QLearningInventoryAgent(config=DEFAULT_RL_CONFIG)
    rewards = agent.train(episodes=60)
    assert len(rewards) == 60
    assert len(agent.q_table) > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "agent.joblib"
        agent.save(str(path))
        loaded = QLearningInventoryAgent.load(str(path))
        assert len(loaded.q_table) == len(agent.q_table)
