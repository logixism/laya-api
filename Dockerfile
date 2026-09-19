FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    # Hugging Face cache lives outside the image so model weights survive
    # restarts instead of re-downloading (~GB) on every container start.
    HF_HOME=/data/hf

# Install dependencies first for layer caching; torch + CUDA wheels are the bulk.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

VOLUME /data/hf

EXPOSE 8000

# Requires the NVIDIA Container Toolkit; run with: docker run --gpus all ...
# LAYA_API_KEY must be provided (-e LAYA_API_KEY=...), the app refuses to start without it.
CMD ["laya-api"]
