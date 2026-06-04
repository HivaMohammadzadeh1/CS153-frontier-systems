# Memex web app — FastAPI + static frontend. Runs migrations on boot, then serves.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 UV_COMPILE_BYTECODE=1
WORKDIR /app

RUN pip install --no-cache-dir uv

# Source first so the local package installs with its code present.
COPY . .
RUN uv sync --frozen --no-dev

# Render injects $PORT. Apply DB migrations (idempotent) before serving.
ENV PORT=8000
CMD ["sh", "-c", "uv run python -m scripts.migrate && uv run uvicorn learning_memory_os.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
