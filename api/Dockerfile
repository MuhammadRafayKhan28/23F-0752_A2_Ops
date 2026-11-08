# Model API image for the cloud (Task 7). The champion model is BAKED IN: the cloud machine has no
# access to our repo/DVC, and an immutable image = exactly reproducible deployment.
# Build from repo root:  docker build -f api/Dockerfile -t <dockerhub-user>/23f-0752_a2_ops-api:1 .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 PORT=8000 \
    MODEL_PATH=/app/models/champion/visionguard.keras MODEL_META_PATH=/app/models/champion/meta.json

WORKDIR /app
COPY requirements-web.txt .
RUN pip install -r requirements-web.txt

RUN useradd --create-home --uid 10001 appuser
COPY serving/ serving/
COPY api/ api/
COPY models/champion/ models/champion/
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"
# 1 worker: each worker holds its own copy of the model in RAM; threads handle concurrency
CMD exec gunicorn -w 1 --threads 4 --timeout 120 -b 0.0.0.0:${PORT} api.app:app
