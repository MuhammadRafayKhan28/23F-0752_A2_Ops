import io

import pytest
from conftest import png_bytes


class FakePredictor:
    model_name, model_version, threshold, img_size = "fake", "0", 0.5, 224

    def predict_bytes(self, data):
        return {"label": "Real", "confidence": 0.8, "prob_ai": 0.2, "threshold": 0.5,
                "model_name": "fake", "model_version": "0", "latency_ms": 1.0}


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    from api.app import create_app
    return create_app(FakePredictor()).test_client()


def test_api_requires_key(api):
    r = api.post("/predict", data={"file": (io.BytesIO(png_bytes()), "a.png")})
    assert r.status_code == 401


def test_api_predict(api):
    r = api.post("/predict", headers={"X-API-Key": "secret123"},
                 data={"file": (io.BytesIO(png_bytes()), "a.png")})
    assert r.status_code == 200 and r.get_json()["label"] == "Real"


def test_api_health_public(api):
    assert api.get("/health").get_json()["status"] == "ok"
