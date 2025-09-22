FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制源代码
COPY . .

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV ENV=production
ENV MILVUS_HOST=milvus-standalone
ENV MILVUS_PORT=19530
ENV ARK_API_KEY=1a2088242b224a7fac8949c4b1dcc5a7
ENV ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ENV ARK_MODEL=ep-20241201140014-vbzjz
ENV EMBEDDING_MODEL=bge-m3
ENV EMBEDDING_DIMENSION=1024
ENV LOG_LEVEL=info

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
