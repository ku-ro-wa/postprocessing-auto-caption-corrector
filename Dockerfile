FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1
WORKDIR /app

# Dependencies first, so a source-only change reuses this layer.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project

COPY src ./src
RUN uv sync --locked

ENV PATH="/app/.venv/bin:$PATH" \
    CAPTION_CHECKER_DATA_DIR=/data \
    CAPTION_CHECKER_SECURE_COOKIE=1

EXPOSE 8080
CMD ["caption-checker", "serve", "--host", "0.0.0.0", "--port", "8080"]
