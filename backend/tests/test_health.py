"""
backend/tests/test_health.py
Tests for /health and /ready endpoints (Phase 12).
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture(scope="module")
def client():
    from backend.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_returns_200(client):
    """GET /health must always return 200."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_health_returns_200_or_503(client):
    """GET /api/health returns 200 (when artifacts loaded) or 503 (if not loaded)."""
    r = client.get("/api/health")
    assert r.status_code in (200, 503)
    data = r.json()
    assert "status" in data


def test_ready_returns_200_or_503(client):
    """GET /ready returns 200 when retrieval service is initialized, 503 otherwise."""
    r = client.get("/ready")
    # Either is acceptable depending on test environment (artifacts may not be loaded)
    assert r.status_code in (200, 503)
    data = r.json()
    assert "status" in data
    if r.status_code == 200:
        assert data["status"] == "ready"
        assert "languages" in data
    else:
        assert data["status"] == "not_ready"


def test_health_independent_of_ready():
    """
    /health must return 200 even when retrieval service is NOT initialized.
    This tests that liveness != readiness.
    """
    from backend.main import app
    from fastapi.testclient import TestClient

    with patch("backend.pipeline.retrieval_service.retrieval_service") as mock_rs:
        mock_rs.initialized = False
        with TestClient(app, raise_server_exceptions=False) as c:
            # /health should ALWAYS be 200 regardless of retrieval_service state
            r = c.get("/health")
            assert r.status_code == 200
