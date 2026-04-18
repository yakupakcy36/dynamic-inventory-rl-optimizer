"""Integration tests for Dynamic Inventory RL API endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["policy_loaded"] is True


def test_policy_recommend_endpoint(client):
    payload = {
        "on_hand_inventory": 15,
        "in_transit_pipeline_qty": 0,
        "yesterday_demand": 14
    }
    response = client.post("/policy/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["recommended_order_quantity"] >= 0
    assert len(data["state_bucket"]) == 3
    assert data["total_inventory_position"] == 15


def test_benchmark_compare_endpoint(client):
    response = client.get("/benchmark/compare")
    assert response.status_code == 200
    data = response.json()
    assert data["eval_horizon_days"] == 180
    assert len(data["policies"]) == 3
