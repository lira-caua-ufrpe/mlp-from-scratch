"""
Testes unitários e de integração da API do TEA Monitor
"""
import pytest
from app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"

def test_post_and_get_events(client):
    # Limpa eventos
    client.delete("/api/events")

    # Posta um evento
    event_payload = {
        "type": "HAND_FLAPPING",
        "label": "Flapping de Mãos",
        "confidence": 0.85,
        "severity": "warning",
        "metrics": {"flappingHz": 3.4}
    }
    res_post = client.post("/api/events", json=event_payload)
    assert res_post.status_code == 201

    # Busca eventos
    res_get = client.get("/api/events")
    assert res_get.status_code == 200
    data = res_get.get_json()
    assert data["total"] == 1
    assert data["events"][0]["type"] == "HAND_FLAPPING"
