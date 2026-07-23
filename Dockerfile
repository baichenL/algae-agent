# syntax=docker/dockerfile:1.7

FROM node:22-alpine AS web-builder
WORKDIR /src/web
RUN corepack enable
COPY web/package.json web/pnpm-lock.yaml web/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY web/ ./
RUN pnpm build

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp

WORKDIR /app

# Install Python dependencies before application code so dependency layers remain cacheable.
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt

# Pin the public model revision and copy it to a fixed container-local path.
ARG RAG_BGE_MODEL_ID=BAAI/bge-reranker-v2-m3
ARG RAG_BGE_MODEL_REVISION=953dc6f
ENV RAG_BGE_MODEL_ID=${RAG_BGE_MODEL_ID} \
    RAG_BGE_MODEL_REVISION=${RAG_BGE_MODEL_REVISION}
RUN python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['RAG_BGE_MODEL_ID'], revision=os.environ['RAG_BGE_MODEL_REVISION'], local_dir='/opt/models/bge-reranker-v2-m3')"

COPY app/ ./app/
COPY frontend/ ./frontend/
COPY --from=web-builder /src/web/dist ./web/dist

ENV RAG_BGE_RERANKER_MODEL=/opt/models/bge-reranker-v2-m3 \
    RAG_BGE_LOCAL_FILES_ONLY=true \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN groupadd --system --gid 10001 algae \
    && useradd --system --uid 10001 --gid algae --home-dir /app --shell /usr/sbin/nologin algae \
    && mkdir -p data/logs data/outputs data/raw \
    && chown -R algae:algae /app/data /opt/models

USER algae
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
