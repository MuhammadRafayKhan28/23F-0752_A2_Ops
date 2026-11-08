"""VisionGuard Flask web app (Tasks 4, 5, 8).

All settings come from environment variables (Task 4.4). Debug mode is always off.
Local dev:   flask --app webapp.app run           (debug stays off)
Container:   gunicorn -w 1 -b 0.0.0.0:5000 webapp.app:app
"""
import logging
import os
import time

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from serving.validation import UploadError, validate_upload
from webapp.db import Database

log = logging.getLogger("visionguard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

CAVEAT = ("This verdict is a statistical estimate from a model trained on roughly a thousand web images. "
          "It can be wrong - especially for edited, heavily compressed, very small or painted images, and for "
          "images from generators it never saw. Do not use it as the only evidence about an image.")



def create_app(backend=None, database_url=None, max_mb=None):
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("SECRET_KEY", os.urandom(16).hex()),
        MAX_UPLOAD_BYTES=int(float(max_mb or os.getenv("MAX_UPLOAD_MB", "5")) * 1024 * 1024),
        DATABASE_URL=database_url or os.getenv("DATABASE_URL", "sqlite:///visionguard.db"),
        INFERENCE_MODE=os.getenv("INFERENCE_MODE", "local"),
    )
    # hard limit enforced by Werkzeug before the body is read (+64 KB for multipart overhead)
    app.config["MAX_CONTENT_LENGTH"] = app.config["MAX_UPLOAD_BYTES"] + 64 * 1024

    if backend is None:
        if app.config["INFERENCE_MODE"] == "remote":
            from webapp.backends import RemoteBackend
            backend = RemoteBackend(os.getenv("MODEL_API_URL", ""), os.getenv("MODEL_API_KEY", ""))
        else:
            from webapp.backends import LocalBackend
            backend = LocalBackend()  # model loaded ONCE here, at start-up
    app.backend = backend
    app.db = Database(app.config["DATABASE_URL"])
    log.info("VisionGuard ready: mode=%s db=%s", backend.mode, app.db.engine.url.render_as_string(hide_password=True))

    @app.get("/")
    def index():
        return render_template("index.html", max_mb=app.config["MAX_UPLOAD_BYTES"] // (1024 * 1024))

    @app.post("/predict")
    def predict():
        f = request.files.get("image")
        if f is None or not f.filename:
            raise UploadError("Please choose an image file to upload.")
        data = f.read()  # kept in memory only; never written to disk
        info = validate_upload(f.filename, data, app.config["MAX_UPLOAD_BYTES"])
        t0 = time.time()
        try:
            res = app.backend.predict(info["filename"], data, f.mimetype)
        except RuntimeError as e:
            log.exception("inference failed")
            return render_template("error.html", code=502, title="Model unavailable", message=str(e)), 502
        latency = res.get("latency_ms") or round((time.time() - t0) * 1000, 1)
        row = app.db.add(
            filename=info["filename"], label=res["label"], confidence=res["confidence"], prob_ai=res["prob_ai"],
            threshold=res["threshold"], model_name=res["model_name"], model_version=str(res["model_version"]),
            inference_mode=app.backend.mode, latency_ms=latency, image_format=info["format"],
            image_width=info["width"], image_height=info["height"], file_size_bytes=info["size_bytes"],
            client_ip=request.headers.get("X-Forwarded-For", request.remote_addr or "")[:64],
        )
        log.info("prediction id=%s file=%s label=%s conf=%.3f", row.id, row.filename, row.label, row.confidence)
        return render_template("result.html", r=row, caveat=CAVEAT)

    @app.get("/history")
    def history():
        rows = app.db.recent(int(request.args.get("limit", 100)))
        return render_template("history.html", rows=rows)

    @app.get("/api/history")
    def history_json():
        return jsonify([{c.name: getattr(r, c.name) for c in r.__table__.columns}
                        for r in app.db.recent(int(request.args.get("limit", 100)))])

    @app.get("/health")
    def health():
        db_ok = app.db.ping()
        body = {"status": "ok" if db_ok else "degraded", "database": db_ok,
                "inference_mode": app.backend.mode, "model": app.backend.info()}
        return jsonify(body), 200 if db_ok else 503

    # ---------------- friendly errors
    @app.errorhandler(UploadError)
    def upload_error(e):
        return render_template("error.html", code=e.code, title="Upload rejected", message=str(e)), e.code

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(e):
        mb = app.config["MAX_UPLOAD_BYTES"] // (1024 * 1024)
        return render_template("error.html", code=413, title="File too large",
                               message=f"The upload limit is {mb} MB."), 413

    @app.errorhandler(HTTPException)
    def http_error(e):
        return render_template("error.html", code=e.code, title=e.name, message=e.description), e.code

    @app.errorhandler(Exception)
    def server_error(e):
        log.exception("unhandled error")
        return render_template("error.html", code=500, title="Something went wrong",
                               message="An unexpected error occurred. Please try again."), 500

    return app


# module-level app for gunicorn / `flask --app webapp.app run`; skipped when imported by tests
app = create_app() if os.getenv("VG_NO_AUTOCREATE") != "1" else None

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
