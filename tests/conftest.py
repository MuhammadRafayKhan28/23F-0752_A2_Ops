import io
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

os.environ["VG_NO_AUTOCREATE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeBackend:
    """Stands in for the Keras model so CI needs no TensorFlow."""
    mode = "local"

    def info(self):
        return {"model_name": "fake", "model_version": "0", "threshold": 0.5}

    def predict(self, filename, data, content_type):
        return {"label": "AI-generated", "confidence": 0.9, "prob_ai": 0.9, "threshold": 0.5,
                "model_name": "fake", "model_version": "0", "latency_ms": 1.0}


def png_bytes(size=(64, 48), color=(200, 30, 30)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    from webapp.app import create_app
    app = create_app(backend=FakeBackend(), database_url=f"sqlite:///{tmp_path}/t.db", max_mb=1)
    app.testing = True
    return app.test_client()
