import uuid

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from tests.test_auth import db_available

pytestmark = pytest.mark.skipif(not db_available(), reason="PostgreSQL not running")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def auth_header(client):
    acc = {"email": f"fb_{uuid.uuid4().hex[:8]}@example.com", "username": "fbtester", "password": "Secret123!"}
    token = client.post("/auth/register", json=acc).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_post_and_list_feedback(client, auth_header):
    r = client.post("/feedback", json={"rating": 5, "text": "Great plate reader!"}, headers=auth_header)
    assert r.status_code == 201 and r.json()["username"] == "fbtester"
    body = client.get("/feedback").json()
    assert body["count"] >= 1 and body["items"][0]["text"] == "Great plate reader!"


def test_feedback_requires_login(client):
    assert client.post("/feedback", json={"rating": 4, "text": "hello"}).status_code == 401


def test_feedback_rating_range(client, auth_header):
    r = client.post("/feedback", json={"rating": 6, "text": "too many stars"}, headers=auth_header)
    assert r.status_code == 422


def test_valley_requires_login(client):
    assert client.post("/valley/ask", json={"question": "What is ABS?"}).status_code == 401
