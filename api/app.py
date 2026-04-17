"""FastAPI application for Dynamic Inventory Reinforcement Learning Optimizer."""

from contextlib import asynccontextmanager
from typing import Optional, List
from fastapi import FastAPI, HTTPException, status

from src.config import DEFAULT_RL_CONFIG
from src.environment import InventoryGymEnv
from src.agent import QLearningInventoryAgent
from src.benchmark import evaluate_policy, PolicyEvaluationResult
from api.schemas import (
    WarehouseStateRequest,
    ReplenishmentActionResponse,
    PolicyMetricItem,
    BenchmarkResponse,
    HealthResponse,
)

agent: Optional[QLearningInventoryAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent
    artifact_path = DEFAULT_RL_CONFIG.artifacts_dir / DEFAULT_RL_CONFIG.model_filename
    if artifact_path.exists():
        try:
            agent = QLearningInventoryAgent.load(str(artifact_path))
        except Exception:
            agent = QLearningInventoryAgent(config=DEFAULT_RL_CONFIG)
            agent.train(episodes=1500)
    else:
        agent = QLearningInventoryAgent(config=DEFAULT_RL_CONFIG)
        agent.train(episodes=1500)
    yield


app = FastAPI(
    title="Dynamic Inventory Reinforcement Learning Optimizer API",
    description=(
        "Production supply chain AI service. Solves the stochastic lead-time and Poisson demand "
        "Markov Decision Process (MDP) via tabular Q-learning to minimize asymmetric holding vs. "
        "stockout penalty costs while maximizing customer service level (fill rate)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health():
    return HealthResponse(
        status="healthy",
        policy_loaded=(agent is not None),
        state_actions_learned=len(agent.q_table) if agent else 0,
        discount_factor=DEFAULT_RL_CONFIG.gamma_discount_factor,
    )


@app.post("/policy/recommend", response_model=ReplenishmentActionResponse, tags=["Policy Recommendation"])
def recommend_order(req: WarehouseStateRequest):
    """Recommends optimal order quantity based on current on-hand and pipeline inventory."""
    if agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RL agent not initialized")

    env = InventoryGymEnv(config=DEFAULT_RL_CONFIG)
    state = env._discretize(req.on_hand_inventory, req.in_transit_pipeline_qty, req.yesterday_demand)

    action = agent.select_action(state, greedy=True)
    inv_pos = req.on_hand_inventory + req.in_transit_pipeline_qty

    if action == 0:
        rational = f"Inventory position ({inv_pos}) sufficient against expected lead-time demand."
    else:
        rational = f"Replenishment of {action} units ordered to mitigate stockout penalty under stochastic lead time."

    return ReplenishmentActionResponse(
        recommended_order_quantity=action,
        state_bucket=list(state),
        total_inventory_position=inv_pos,
        policy_type="Q_LEARNING_BELLMAN_OPTIMAL",
        decision_rational=rational,
    )


@app.get("/benchmark/compare", response_model=BenchmarkResponse, tags=["Benchmarking"])
def benchmark_compare():
    """Runs out-of-sample backtest comparing RL policy vs Fixed (s, S) heuristic."""
    if agent is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RL agent not initialized")

    res_rl = evaluate_policy("RL_AGENT", agent=agent, eval_days=180, seed=999)
    res_ss = evaluate_policy("FIXED_s_S", eval_days=180, seed=999)
    res_jit = evaluate_policy("NAIVE_JIT", eval_days=180, seed=999)

    service_diff = res_rl.service_level_fill_rate_pct - res_ss.service_level_fill_rate_pct
    stockout_reduction = ((res_ss.stockout_days - res_rl.stockout_days) / max(res_ss.stockout_days, 1)) * 100.0

    items = [
        PolicyMetricItem(**res_rl.__dict__),
        PolicyMetricItem(**res_ss.__dict__),
        PolicyMetricItem(**res_jit.__dict__),
    ]

    return BenchmarkResponse(
        eval_horizon_days=180,
        policies=items,
        service_level_improvement_pct=round(service_diff, 1),
        stockout_days_reduction_pct=round(stockout_reduction, 1),
    )
