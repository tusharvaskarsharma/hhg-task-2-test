import os
import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_frontend_has_no_toggle_labels():
    app_jsx_path = os.path.join(os.path.dirname(__file__), "../../frontend/src/App.jsx")
    with open(app_jsx_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    forbidden_labels = [
        "Fast extractive mode",
        "Grounded SLM mode",
        "Generate grounded answer",
        "isGenerateEnabled",
        "setIsGenerateEnabled",
        "<input type=\"checkbox\""
    ]
    for label in forbidden_labels:
        assert label not in content, f"Forbidden label '{label}' found in App.jsx"

def test_cors_allows_127_0_0_1():
    response = client.options("/api/query", headers={
        "Origin": "http://127.0.0.1:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"
    })
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://127.0.0.1:3000"

def test_cors_allows_localhost():
    response = client.options("/api/query", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type"
    })
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
