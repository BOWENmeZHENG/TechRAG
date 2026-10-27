# syntax=docker/dockerfile:1

# Build stage: install the locked dependencies, then build the index from data/ so the image
# ships with the index and the embedding model already in place.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    HF_HOME=/opt/hf
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

COPY data ./data
RUN .venv/bin/techrag-ingest data --index-dir .index

# Runtime stage: only the virtualenv, the index and the cached model.
FROM python:3.12-slim
ENV PATH=/app/.venv/bin:$PATH \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app

RUN useradd --system --no-create-home techrag
COPY --from=build /app/.venv ./.venv
COPY --from=build /app/.index ./.index
COPY --from=build /opt/hf /opt/hf

USER techrag
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

# Listen on all interfaces so the published port is reachable from the host.
ENTRYPOINT ["techrag-serve", "--host", "0.0.0.0"]
