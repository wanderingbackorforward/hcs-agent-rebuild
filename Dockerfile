# syntax=docker/dockerfile:1.6
# Multi-stage build for hcs-agent-rebuild.
# Build:  docker build -t hcs-agent-rebuild:dev .
# Run:    docker run --rm -p 8000:8000 --env-file .env hcs-agent-rebuild:dev

# ---------- builder stage ----------
FROM python:3.12-slim AS builder

# Install build deps for chromadb / rank-bm25 / etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source.
COPY . .

# ---------- runtime stage ----------
FROM python:3.12-slim AS runtime

# Non-root user for the app process.
RUN groupadd --system --gid 1001 hcs \
    && useradd --system --uid 1001 --gid hcs --create-home --shell /bin/bash hcs

WORKDIR /app

# Copy installed packages and source from builder.
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build /app

# Pre-create writable data dir for SQLite / Chroma persistence.
RUN mkdir -p /app/data /app/logs /app/.chroma \
    && chown -R hcs:hcs /app

USER hcs

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

# Health check: hit the FastAPI docs endpoint (cheap, no LLM call).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/docs', timeout=3).read()" \
        || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
