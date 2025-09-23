FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements-basic.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements-basic.txt

# 复制源代码
COPY . .

# 设置执行权限
RUN chmod +x start.sh

# 环境变量通过 .env 文件或 docker-compose 环境变量传入
# 包括但不限于：
# PYTHONPATH, PYTHONUNBUFFERED, PORT, LOG_LEVEL
# GLM_API_KEY, GLM_BASE_URL, ARK_API_KEY, ARK_MODEL 等
ENV PYTHONPATH=/app

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["./start.sh"]
