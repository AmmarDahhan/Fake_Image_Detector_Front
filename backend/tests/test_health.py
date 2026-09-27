"""Tests for GET /health."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.schemas.analysis import HealthResponse


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_matches_response_schema(client: TestClient) -> None:
    payload = client.get("/health").json()
    assert set(payload) == {"status"}
    assert HealthResponse.model_validate(payload).status == "ok"


def test_health_accepts_a_plain_get(client: TestClient) -> None:
    """No body, no headers, no upload - the probe must stay dependency-free."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
