"""Pydantic schemas for Dynamic Inventory RL Optimizer API."""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class WarehouseStateRequest(BaseModel):
    on_hand_inventory: int = Field(default=35, ge=-100, le=500, description="Current on-hand warehouse inventory units")
    in_transit_pipeline_qty: int = Field(default=20, ge=0, le=500, description="Units already ordered and in transit")
    yesterday_demand: int = Field(default=12, ge=0, le=200, description="Realized customer demand yesterday")


class ReplenishmentActionResponse(BaseModel):
    recommended_order_quantity: int
    state_bucket: List[int]
    total_inventory_position: int
    policy_type: str
    decision_rational: str


class PolicyMetricItem(BaseModel):
    policy_name: str
    total_cost_usd: float
    holding_cost_usd: float
    stockout_penalty_usd: float
    order_cost_usd: float
    service_level_fill_rate_pct: float
    stockout_days: int


class BenchmarkResponse(BaseModel):
    eval_horizon_days: int
    policies: List[PolicyMetricItem]
    service_level_improvement_pct: float
    stockout_days_reduction_pct: float


class HealthResponse(BaseModel):
    status: str
    policy_loaded: bool
    state_actions_learned: int
    discount_factor: float
