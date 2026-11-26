import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from src.api.db import engine
from src.api.main import app


def db_available() -> bool:
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not db_available(), reason="PostgreSQL not running")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def account():
    return {"email": f"test_{uuid.uuid4().hex[:8]}@example.com", "username": "tester", "password": "Secret123!"}


def test_register_login_me(client, account):
    r = client.post("/auth/register", json=account)
    assert r.status_code == 201 and r.json()["access_token"]

    r = client.post("/auth/login", json={"email": account["email"], "password": account["password"]})
    token = r.json()["access_token"]
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200 and me.json()["email"] == account["email"]


def test_duplicate_email(client, account):
    assert client.post("/auth/register", json=account).status_code == 409


def test_wrong_password(client, account):
    r = client.post("/auth/login", json={"email": account["email"], "password": "wrong-password"})
    assert r.status_code == 401


def test_me_without_token(client):
    assert client.get("/auth/me").status_code == 401


def test_weak_password_rejected(client):
    r = client.post("/auth/register", json={"email": "x@example.com", "username": "xx1", "password": "123"})
    assert r.status_code == 422