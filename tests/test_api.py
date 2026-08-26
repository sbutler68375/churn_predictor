"""Automated tests for the /predict endpoint. Uses FastAPI's TestClient, so
these run fully in-process -- no uvicorn server, no Postgres connection."""

from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "tenant_id": "0aa66501-3e08-4f21-9cb5-277a89644d3a",
    "contract_value": 8357.40,
    "events_last_7d": 179,
    "events_last_30d": 1067,
    "events_last_90d": 4201,
    "support_tickets_last_30d": 162,
    "subscription_tier": "Starter",
}


def test_predict_valid_payload_returns_200():
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200

    body = response.json()
    assert body["tenant_id"] == VALID_PAYLOAD["tenant_id"]
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in {"low", "medium", "high"}


def test_predict_missing_field_returns_422():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "contract_value"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_wrong_type_returns_422():
    payload = {**VALID_PAYLOAD, "events_last_7d": "a lot"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_empty_payload_returns_422():
    response = client.post("/predict", json={})
    assert response.status_code == 422
