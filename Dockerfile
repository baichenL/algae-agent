FROM node:22-alpine AS web-builder
WORKDIR /src/web
RUN corepack enable
COPY web/package.json web/pnpm-lock.yaml web/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY web/ ./
RUN pnpm build

# 使用轻量 Python 镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app
ENV PYTHONUNBUFFERED=1
# 拷贝代码和依赖文件
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY --from=web-builder /src/web/dist ./web/dist
COPY requirements.txt .
# 安装依赖
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
RUN mkdir -p data/logs data/outputs data/raw
EXPOSE 8000
# 设置容器默认命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
