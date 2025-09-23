# 多阶段构建 - 算法服务
FROM python:3.11-slim AS base

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    wget \
    git \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 升级pip和安装基础工具
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 依赖安装阶段
FROM base AS deps

# 复制依赖文件
COPY requirements*.txt ./

# 安装Python依赖（使用缓存优化）
RUN pip install --no-cache-dir -r requirements.txt

# 生产阶段
FROM base AS production

# 复制已安装的依赖
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# 复制源代码
COPY . .

# 创建数据目录
RUN mkdir -p /app/data /app/logs /app/models

# 设置执行权限
RUN chmod +x start.sh start_enhanced.sh

# 创建非root用户
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# 环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["./start_enhanced.sh"]
