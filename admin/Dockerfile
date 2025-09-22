FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt* ./

# 安装Python依赖
RUN pip install --no-cache-dir flask gunicorn psycopg2-binary redis flask-login flask-sqlalchemy flask-cors pandas requests

# 复制源代码
COPY . .

# 环境变量通过 .env 文件或 docker-compose 环境变量传入
# 包括但不限于：
# FLASK_ENV, FLASK_DEBUG, PORT, DATABASE_URL, REDIS_URL 等

# 暴露端口
EXPOSE 5001

# 启动命令
CMD ["python", "app.py"]
