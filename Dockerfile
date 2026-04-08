````dockerfile
# =============================================================================
# AI Medical Triage & Clinical Documentation Environment
# Multi-stage Dockerfile — Hugging Face Spaces compatible (port 7860)
# =============================================================================

FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


FROM python:3.11-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PORT=7860 \
    HOST=0.0.0.0 \
    WORKERS=1 \
    LOG_LEVEL=info \
    ENV=production \
    SESSION_TTL_SECONDS=1800 \
    MAX_SESSIONS=100 \
    RATE_LIMIT_RPM=60 \
    RATE_LIMITING_ENABLED=true \
    MAX_LEADERBOARD_ENTRIES=500 \
    ANTHROPIC_API_KEY=""

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

WORKDIR /app

RUN groupadd --gid 1001 appgroup \
 && useradd --uid 1001 --gid appgroup --shell /bin/bash --create-home appuser

COPY --chown=appuser:appgroup data/        /app/data/
COPY --chown=appuser:appgroup config/      /app/config/
COPY --chown=appuser:appgroup env/         /app/env/
COPY --chown=appuser:appgroup graders/     /app/graders/
COPY --chown=appuser:appgroup server/      /app/server/
COPY --chown=appuser:appgroup agents/      /app/agents/
COPY --chown=appuser:appgroup inference.py /app/inference.py
COPY --chown=appuser:appgroup openenv.yaml /app/openenv.yaml
COPY --chown=appuser:appgroup README.md    /app/README.md

EXPOSE 7860

USER appuser

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=15s \
    --retries=3 \
    CMD curl -fsSL "http://localhost:${PORT}/health" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('status')=='ok' else 1)" \
        || exit 1

CMD ["uvicorn", \
     "server.api_server:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "30", \
     "--log-level", "info", \
     "--no-access-log"]
````