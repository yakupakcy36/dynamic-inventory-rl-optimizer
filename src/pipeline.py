"""CLI runner for training Q-Learning agent and benchmarking against (s, S) heuristic."""

import argparse
from pathlib import Path
from src.config import DEFAULT_RL_CONFIG
from src.agent import QLearningInventoryAgent
from src.benchmark import evaluate_policy


def run_pipeline(episodes: int = 2500, save_artifact: bool = True):
    print("=" * 68)
    print("DYNAMIC INVENTORY RL OPTIMIZER: Q-LEARNING MDP BENCHMARK")
    print("=" * 68)

    print(f"[1/3] Training Tabular Q-Learning Agent across {episodes} episodes...")
    agent = QLearningInventoryAgent(config=DEFAULT_RL_CONFIG)
    rewards = agent.train(episodes=episodes)
    print(f"      Training complete. Unique state-action pairs explored: {len(agent.q_table)}")
    print(f"      Initial episode mean reward: {sum(rewards[:50])/50:.1f} | Final episode mean reward: {sum(rewards[-50:])/50:.1f}")

    print("\n[2/3] Simulating 180-Day Supply Chain Operations across Policies:")
    res_rl = evaluate_policy("RL_AGENT", agent=agent, eval_days=180, seed=999)
    res_ss = evaluate_policy("FIXED_s_S", eval_days=180, seed=999)
    res_jit = evaluate_policy("NAIVE_JIT", eval_days=180, seed=999)

    print(f"  {'Policy Name':18s} | {'Total Cost ($)':14s} | {'Holding ($)':11s} | {'Stockout ($)':12s} | {'Fill Rate (%)':13s} | {'Stockout Days':13s}")
    print("  " + "-" * 95)
    for r in [res_rl, res_ss, res_jit]:
        print(f"  {r.policy_name:18s} | ${r.total_cost_usd:12.2f} | ${r.holding_cost_usd:9.2f} | ${r.stockout_penalty_usd:10.2f} | {r.service_level_fill_rate_pct:11.1f}% | {r.stockout_days:13d}")

    cost_savings = ((res_ss.total_cost_usd - res_rl.total_cost_usd) / res_ss.total_cost_usd) * 100.0
    print(f"\n[3/3] RL Agent achieved {cost_savings:+.1f}% cost reduction vs standard (s, S) heuristic")

    if save_artifact:
        out_path = DEFAULT_RL_CONFIG.artifacts_dir / DEFAULT_RL_CONFIG.model_filename
        print(f"[Artifact] Serializing policy weights to: {out_path}")
        agent.save(str(out_path))

    print("\n[SUCCESS] Inventory RL optimization pipeline finished successfully.")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inventory RL CLI")
    parser.add_argument("--episodes", type=int, default=2500, help="Training episodes")
    args = parser.parse_args()
    run_pipeline(episodes=args.episodes)
