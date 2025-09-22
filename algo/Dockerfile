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

# 环境变量通过 .env 文件或 docker-compose 环境变量传入
# 包括但不限于：
# PYTHONPATH, PYTHONUNBUFFERED, PORT, LOG_LEVEL
# GLM_API_KEY, GLM_BASE_URL, ARK_API_KEY, ARK_MODEL 等

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
