"""VisionGuard model API (Task 7) - deployed to the cloud; the local Flask UI calls it (Task 8).

POST /predict   multipart field 'file', header 'X-API-Key: <key>'  -> JSON verdict
GET  /health    public liveness/readiness probe
"""
import hmac
import json
import logging
import os
import time

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from serving.validation import UploadError, validate_upload

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("visionguard-api")


def create_app(predictor=None):
    app = Flask(__name__)
    max_bytes = int(float(os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024)
    app.config["MAX_CONTENT_LENGTH"] = max_bytes + 64 * 1024
    api_key = os.getenv("API_KEY", "")
    if not api_key:
        log.warning("API_KEY is not set - /predict will reject every request")
    if predictor is None:
        from serving.inference import predictor_from_env
        predictor = predictor_from_env()  # loaded once at start-up
    started = time.time()

    @app.get("/")
    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "service": "android_A2_Ops visionguard-api",
                        "model_name": predictor.model_name, "model_version": predictor.model_version,
                        "threshold": predictor.threshold, "img_size": predictor.img_size,
                        "uptime_s": round(time.time() - started)})

    @app.post("/predict")
    def predict():
        sent = request.headers.get("X-API-Key", "")
        if not api_key or not hmac.compare_digest(sent, api_key):
            return jsonify({"error": "missing or invalid API key"}), 401
        f = request.files.get("file")
        if f is None:
            raise UploadError("multipart field 'file' is required")
        data = f.read()
        info = validate_upload(f.filename or "upload", data, max_bytes)
        res = predictor.predict_bytes(data)
        res.update(filename=info["filename"], width=info["width"], height=info["height"])
        log.info("predict %s", json.dumps(res))  # visible in `docker logs` / cloud log viewer
        return jsonify(res)

    @app.errorhandler(UploadError)
    def bad_upload(e):
        return jsonify({"error": str(e)}), e.code

    @app.errorhandler(HTTPException)
    def http_err(e):
        return jsonify({"error": e.description}), e.code

    return app


app = create_app() if os.getenv("VG_NO_AUTOCREATE") != "1" else None
