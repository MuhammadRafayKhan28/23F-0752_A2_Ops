# VisionGuard web app (Task 6.2). Build from repo root: docker build -f webapp/Dockerfile .
# Order matters for the layer cache: requirements change rarely, source changes often. Installing
# dependencies BEFORE copying the source means editing app.py rebuilds only the last small layers.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# 1) dependencies (cached layer) - tensorflow-cpu, not the CUDA build: ~0.6 GB instead of ~2.5 GB
COPY requirements-web.txt .
RUN pip install -r requirements-web.txt

# 2) unprivileged user
RUN useradd --create-home --uid 10001 appuser

# 3) source code (changes most often -> last). No model and no secrets in the image:
#    the model is mounted read-only by compose, secrets arrive as env vars at runtime.
COPY serving/ serving/
COPY webapp/ webapp/
USER appuser

EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=4)"
CMD ["gunicorn", "-w", "1", "--threads", "4", "--timeout", "120", "-b", "0.0.0.0:5000", "webapp.app:app"]
