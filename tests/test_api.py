from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.config import settings
from src.api.main import app

pytestmark = pytest.mark.skipif(not Path(settings.det_weights).exists(),
                                reason="model weights not available")
TEST_IMG = Path("data/raw/test/images/1001.jpg")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # 'with' runs the lifespan -> loads the models once
        yield c


def jpg_bytes(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["models_loaded"] is True


def test_predict_real_image(client):
    r = client.post("/predict", files={"file": ("car.jpg", TEST_IMG.read_bytes(), "image/jpeg")})
    body = r.json()
    assert r.status_code == 200 and body["num_plates"] >= 1
    assert {"plate_text", "bbox", "uncertain"} <= body["detections"][0].keys()


def test_predict_no_plate(client):
    black = jpg_bytes(np.zeros((480, 640, 3), np.uint8))
    r = client.post("/predict", files={"file": ("black.jpg", black, "image/jpeg")})
    assert r.status_code == 200 and r.json()["success"] is False


def test_wrong_content_type(client):
    r = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_fake_image(client):
    r = client.post("/predict", files={"file": ("fake.jpg", b"not an image", "image/jpeg")})
    assert r.status_code == 400


def test_too_large(client):
    big = b"0" * int((settings.max_upload_mb + 1) * 1024 * 1024)
    r = client.post("/predict", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413