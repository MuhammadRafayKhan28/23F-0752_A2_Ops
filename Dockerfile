# VisionGuard Web Application Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MODEL_PATH=/app/models/champion/visionguard.keras \
    DATABASE_URL=sqlite:////tmp/visionguard.db

WORKDIR /app

# 1) Dependencies
COPY requirements-web.txt .
RUN pip install --no-cache-dir -r requirements-web.txt

# 2) Unprivileged user
RUN useradd --create-home --uid 10001 appuser

# 3) Source code and baked champion model
COPY serving/ serving/
COPY webapp/ webapp/
COPY models/ models/

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 5000 7860 10000

# Bind dynamically to whatever $PORT cloud assigns (Render=10000, Hugging Face=7860, Local=5000)
CMD ["sh", "-c", "gunicorn -w 1 --threads 2 --timeout 180 -b 0.0.0.0:${PORT:-5000} webapp.app:app"]
