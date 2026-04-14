"""Q-Learning Tabular Reinforcement Learning Agent for Inventory Control."""

import os
from collections import defaultdict
from typing import Dict, Tuple, List
import joblib
import numpy as np
from src.config import InventoryRLConfig, DEFAULT_RL_CONFIG
from src.environment import InventoryGymEnv


class QLearningInventoryAgent:
    """Tabular Q-Learning agent with Bellman TD updates and state-action mapping."""

    def __init__(self, config: InventoryRLConfig = DEFAULT_RL_CONFIG):
        self.config = config
        self.actions = [0, 5, 10, 15, 20, 25, 30, 40]
        # Q-table: state tuple -> numpy array of Q-values for each action
        self.q_table: Dict[Tuple[int, int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.actions), dtype=float)
        )
        self.epsilon = config.epsilon_start

    def select_action(self, state: Tuple[int, int, int], greedy: bool = False) -> int:
        """Epsilon-greedy exploration during training, greedy argmax during inference."""
        if not greedy and np.random.rand() < self.epsilon:
            action_idx = int(np.random.randint(len(self.actions)))
        else:
            q_vals = self.q_table[state]
            action_idx = int(np.argmax(q_vals))
        return self.actions[action_idx]

    def update(
        self,
        state: Tuple[int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int],
        done: bool,
    ) -> float:
        """Bellman equation temporal difference update."""
        action_idx = self.actions.index(action)
        current_q = self.q_table[state][action_idx]

        if done:
            target = reward
        else:
            target = reward + self.config.gamma_discount_factor * float(np.max(self.q_table[next_state]))

        td_error = target - current_q
        self.q_table[state][action_idx] += self.config.alpha_learning_rate * td_error
        return float(abs(td_error))

    def decay_epsilon(self) -> None:
        """Decays exploration parameter after each training episode."""
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def train(self, episodes: int = 2500) -> List[float]:
        """Trains agent across repeated simulated warehouse episodes."""
        env = InventoryGymEnv(config=self.config)
        episode_rewards: List[float] = []

        for ep in range(episodes):
            state = env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                action = self.select_action(state, greedy=False)
                res = env.step(action)
                self.update(state, action, res.reward, res.next_state, res.done)
                state = res.next_state
                ep_reward += res.reward
                done = res.done

            self.decay_epsilon()
            episode_rewards.append(ep_reward)

        return episode_rewards

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Convert defaultdict to regular dict for clean pickling
        serializable_data = {
            "q_table": dict(self.q_table),
            "config": self.config,
            "actions": self.actions,
        }
        joblib.dump(serializable_data, filepath)

    @classmethod
    def load(cls, filepath: str) -> "QLearningInventoryAgent":
        data = joblib.load(filepath)
        agent = cls(config=data["config"])
        agent.q_table = defaultdict(lambda: np.zeros(len(agent.actions), dtype=float), data["q_table"])
        agent.actions = data["actions"]
        agent.epsilon = 0.0  # Production serving mode
        return agent
