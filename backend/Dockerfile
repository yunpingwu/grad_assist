# 可选的后端镜像：CPU 运行（torch 在 Linux 上走 pyproject 的 pytorch-cpu 索引）。
# 模型文件不入镜像，运行时通过 MODEL_CACHE_DIR 挂载到 /models（见 docker/docker-compose.yml 的 app profile）。
FROM ghcr.io/astral-sh/uv:0.11.16 AS uv

FROM python:3.13-slim

# torch CPU 需要 libgomp（OpenMP）；curl 供 healthcheck 使用
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 先装依赖，利用 Docker 层缓存；--no-install-project 因为 pyproject 没有 build-system
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY app ./app

EXPOSE 8000

CMD ["python", "-m", "app.main"]
