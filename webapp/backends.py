"""Where predictions come from: the local Keras model (Task 4) or the cloud model API (Task 7/8)."""
import requests


class LocalBackend:
    mode = "local"

    def __init__(self):
        from serving.inference import predictor_from_env  # TensorFlow imported only in local mode
        self.p = predictor_from_env()

    def info(self):
        return {"model_name": self.p.model_name, "model_version": self.p.model_version,
                "threshold": self.p.threshold}

    def predict(self, filename: str, data: bytes, content_type: str) -> dict:
        return self.p.predict_bytes(data)


class RemoteBackend:
    mode = "remote"

    def __init__(self, url: str, api_key: str, timeout: float = 30):
        if not url:
            raise RuntimeError("INFERENCE_MODE=remote needs MODEL_API_URL")
        self.url = url.rstrip("/")
        self.headers = {"X-API-Key": api_key}
        self.timeout = timeout

    def info(self):
        try:
            r = requests.get(f"{self.url}/health", timeout=10)
            return r.json()
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    def predict(self, filename: str, data: bytes, content_type: str) -> dict:
        r = requests.post(f"{self.url}/predict", headers=self.headers, timeout=self.timeout,
                          files={"file": (filename, data, content_type or "application/octet-stream")})
        if r.status_code != 200:
            try:
                msg = r.json().get("error", r.text)
            except ValueError:
                msg = r.text[:200]
            raise RuntimeError(f"Model API returned {r.status_code}: {msg}")
        return r.json()
