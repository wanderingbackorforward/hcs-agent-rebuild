"""API auth + rate-limit tests.

Uses FastAPI TestClient. Sets API_KEYS env to enable auth mode so we can
test 401 and 429 paths. Tears down by unsetting at the end.
"""
import os
import pytest


@pytest.fixture
def auth_enabled_app(monkeypatch):
    monkeypatch.setenv("API_KEYS", "test-key-1,test-key-2")
    monkeypatch.setenv("RATE_LIMIT_MAX", "2")
    monkeypatch.setenv("RATE_LIMIT_WINDOW", "60")
    # Force re-import so api/auth.py reads the new env at module load.
    import importlib
    import api.auth
    importlib.reload(api.auth)
    from app import create_app
    app = create_app()
    yield app


@pytest.fixture
def client(auth_enabled_app):
    from fastapi.testclient import TestClient
    return TestClient(auth_enabled_app)


def test_no_api_key_returns_401(client):
    r = client.post("/chat", json={"message": "hello"})
    assert r.status_code == 401
    assert "X-API-Key" in r.json()["detail"]


def test_wrong_api_key_returns_401(client):
    r = client.post(
        "/chat",
        json={"message": "hello"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert r.status_code == 401


def test_correct_api_key_routes_to_agent(client, monkeypatch):
    """The full path (auth + rate-limit + agent) should at least not 401/429."""
    # Monkeypatch the handler to avoid hitting the real LLM.
    from api import chat_handler
    async def fake_stream(user_input, session_id=None):
        for tok in ["OK"]:
            yield tok
    monkeypatch.setattr(chat_handler, "process_user_input_stream", fake_stream)

    r = client.post(
        "/chat",
        json={"message": "hello"},
        headers={"X-API-Key": "test-key-1"},
    )
    # 200 OK (or 500 only if agent really failed), but never 401/429.
    assert r.status_code not in (401, 429)


def test_rate_limit_returns_429_after_threshold(client, monkeypatch):
    """After RATE_LIMIT_MAX requests, the next one returns 429."""
    from api import chat_handler
    async def fake_stream(user_input, session_id=None):
        for tok in ["OK"]:
            yield tok
    monkeypatch.setattr(chat_handler, "process_user_input_stream", fake_stream)

    headers = {"X-API-Key": "test-key-2"}
    # RATE_LIMIT_MAX=2: first 2 succeed, 3rd gets 429.
    for _ in range(2):
        r = client.post("/chat", json={"message": "hi"}, headers=headers)
        assert r.status_code == 200
    r = client.post("/chat", json={"message": "hi"}, headers=headers)
    assert r.status_code == 429
    assert "Retry-After" in r.headers
