# VoiceHelper 环境配置指南

## 📋 配置概述

本指南整合了所有环境配置相关文档，提供从开发到生产的完整环境配置方案。

## 🔑 核心环境变量

### 必需配置（生产环境必须）

#### 豆包 API 配置

```bash
# ⚠️ 必须配置，否则AI功能无法使用

ARK_API_KEY=your_ark_api_key_here
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL=ep-20241201140014-vbzjz

# 可选：OpenAI 备用模型

OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
```text

#### 数据库配置

```bash
# PostgreSQL 主数据库

DATABASE_URL=postgres://chatbot:chatbot123@localhost:5432/chatbot?sslmode=disable

# Redis 缓存数据库

REDIS_URL=redis://localhost:6379


MILVUS_HOST=localhost
MILVUS_PORT=19530
```text

### 可选配置

#### 应用配置

```bash
# 服务端口

PORT=8080                    # 后端服务端口
ALGO_PORT=8000              # 算法服务端口
FRONTEND_PORT=3000          # 前端服务端口

# 日志级别

LOG_LEVEL=info              # debug, info, warn, error

# 运行模式

GIN_MODE=release            # debug, release
NODE_ENV=production         # development, production
```text

#### 安全配置

```bash
# JWT 密钥（生产环境必须修改）

JWT_SECRET=11e8ba7f6690ebbe069afd0bf43cfcc3ad5bacefb1fe816c04f90bb022749995

# 管理员密钥

ADMIN_SECRET_KEY=b29d40c1ea82b3331a76b7479526e67fcbf7e5d09719ddf8e6c607dae81756fa

# 数据库密码（生产环境必须修改）

POSTGRES_PASSWORD=chatbot123
REDIS_PASSWORD=redis123
```text

## 🏗️ 环境分类配置

### 开发环境 (Development)

```bash
# .env.development

NODE_ENV=development
GIN_MODE=debug
LOG_LEVEL=debug

# 开发数据库

DATABASE_URL=postgres://chatbot:chatbot123@localhost:5432/chatbot_dev?sslmode=disable
REDIS_URL=redis://localhost:6379/1

# 开发API配置

ARK_API_KEY=dev_api_key
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL=ep-20241201140014-vbzjz

# 开发服务地址

NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_WS_URL=ws://localhost:8080
ALGO_SERVICE_URL=http://localhost:8000
```text

### 测试环境 (Testing)

```bash
# .env.testing

NODE_ENV=test
GIN_MODE=release
LOG_LEVEL=warn

# 测试数据库

DATABASE_URL=postgres://chatbot:chatbot123@localhost:5432/chatbot_test?sslmode=disable
REDIS_URL=redis://localhost:6379/2

# 测试API配置

ARK_API_KEY=test_api_key
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL=ep-20241201140014-vbzjz

# 测试服务地址

NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_WS_URL=ws://localhost:8080
ALGO_SERVICE_URL=http://localhost:8000
```text

### 生产环境 (Production)

```bash
# .env.production

NODE_ENV=production
GIN_MODE=release
LOG_LEVEL=info

# 生产数据库（使用强密码）

DATABASE_URL=postgres://chatbot:STRONG_PASSWORD@prod-db:5432/chatbot?sslmode=require
REDIS_URL=redis://:STRONG_PASSWORD@prod-redis:6379

# 生产API配置

ARK_API_KEY=prod_api_key
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL=ep-20241201140014-vbzjz

# 生产服务地址

NEXT_PUBLIC_API_URL=https://api.voicehelper.com
NEXT_PUBLIC_WS_URL=wss://api.voicehelper.com
ALGO_SERVICE_URL=http://algo-service:8000
```text

## 🐳 Docker 环境配置

### Docker Compose 配置

```yaml
# docker-compose.yml

version: '3.8'
services:
  backend:
    build: ./backend
    environment:
      - DATABASE_URL=postgres://chatbot:chatbot123@postgres:5432/chatbot?sslmode=disable

      - REDIS_URL=redis://redis:6379
      - ALGO_SERVICE_URL=http://algo-service:8000

      - ARK_API_KEY=${ARK_API_KEY}
      - ARK_BASE_URL=${ARK_BASE_URL}

      - ARK_MODEL=${ARK_MODEL}
    depends_on:
      - postgres

      - redis
      - algo-service

  algo-service:
    build: ./algo
    environment:

      - MILVUS_PORT=19530
      - ARK_API_KEY=${ARK_API_KEY}

      - ARK_BASE_URL=${ARK_BASE_URL}
      - ARK_MODEL=${ARK_MODEL}

    depends_on:

  frontend:
    build: ./frontend
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8080

      - NEXT_PUBLIC_WS_URL=ws://localhost:8080
    depends_on:
      - backend

```text

### 环境变量文件

```bash
# .env

ARK_API_KEY=your_ark_api_key_here
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL=ep-20241201140014-vbzjz

# 数据库配置

POSTGRES_PASSWORD=chatbot123
REDIS_PASSWORD=redis123

# 服务配置

LOG_LEVEL=info
GIN_MODE=release
NODE_ENV=production
```text

## ☸️ Kubernetes 环境配置

### ConfigMap 配置

```yaml
# k8s/configmap.yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: voicehelper-config
data:
  LOG_LEVEL: "info"
  GIN_MODE: "release"
  NODE_ENV: "production"
  ARK_BASE_URL: "https://ark.cn-beijing.volces.com/api/v3"
  ARK_MODEL: "ep-20241201140014-vbzjz"
  MILVUS_PORT: "19530"
```text

### Secret 配置

```yaml
# k8s/secrets.yaml

apiVersion: v1
kind: Secret
metadata:
  name: voicehelper-secrets
type: Opaque
data:
  ARK_API_KEY: <base64-encoded-api-key>
  DATABASE_URL: <base64-encoded-database-url>
  REDIS_URL: <base64-encoded-redis-url>
  JWT_SECRET: <base64-encoded-jwt-secret>
  ADMIN_SECRET_KEY: <base64-encoded-admin-secret>
```text

## 🔧 服务特定配置

### 后端服务 (Go)

```bash
# 后端服务环境变量

PORT=8080
GIN_MODE=release
DATABASE_URL=postgres://chatbot:chatbot123@postgres:5432/chatbot?sslmode=disable
REDIS_URL=redis://redis:6379
ALGO_SERVICE_URL=http://algo-service:8000
JWT_SECRET=your_jwt_secret_here
ADMIN_SECRET_KEY=your_admin_secret_here
LOG_LEVEL=info
```text

### 算法服务 (Python)

```bash
# 算法服务环境变量

PORT=8000
ENV=production
MILVUS_PORT=19530
ARK_API_KEY=your_ark_api_key_here
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL=ep-20241201140014-vbzjz
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIMENSION=1024
LOG_LEVEL=info
```text

### 前端服务 (Next.js)

```bash
# 前端服务环境变量

NODE_ENV=production
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_WS_URL=ws://localhost:8080
PORT=3000
HOSTNAME=0.0.0.0
```text

## 📊 性能优化配置

### 数据库优化

```bash
# PostgreSQL 优化

POSTGRES_SHARED_BUFFERS=256MB
POSTGRES_EFFECTIVE_CACHE_SIZE=1GB
POSTGRES_MAINTENANCE_WORK_MEM=64MB
POSTGRES_CHECKPOINT_COMPLETION_TARGET=0.9
POSTGRES_WAL_BUFFERS=16MB
POSTGRES_DEFAULT_STATISTICS_TARGET=100
```text

### Redis 优化

```bash
# Redis 优化

REDIS_MAXMEMORY=1gb
REDIS_MAXMEMORY_POLICY=allkeys-lru
REDIS_SAVE=900 1 300 10 60 10000
REDIS_TCP_KEEPALIVE=60
```text

### 应用优化

```bash
# 并发配置

WORKER_PROCESSES=4
MAX_CONNECTIONS=100
CONNECTION_POOL_SIZE=20
REQUEST_TIMEOUT=30s
KEEP_ALIVE_TIMEOUT=65s
```text

## 🔒 安全配置

### 生产环境安全清单

```bash
# 必须修改的默认配置

POSTGRES_PASSWORD=STRONG_PASSWORD_HERE
REDIS_PASSWORD=STRONG_PASSWORD_HERE
JWT_SECRET=STRONG_JWT_SECRET_HERE
ADMIN_SECRET_KEY=STRONG_ADMIN_SECRET_HERE

# 网络安全

SSL_MODE=require
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem

# 访问控制

ALLOWED_ORIGINS=https://voicehelper.com,https://www.voicehelper.com
CORS_ENABLED=true
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60s
```text

### 密钥管理

```bash
# 使用环境变量文件（不要提交到版本控制）

echo "ARK_API_KEY=your_actual_api_key" > .env.local
echo "DATABASE_URL=postgres://user:pass@host:port/db" >> .env.local
echo "JWT_SECRET=your_jwt_secret" >> .env.local

# 添加到 .gitignore

echo ".env.local" >> .gitignore
echo ".env.production" >> .gitignore
```text

## 🚨 故障排除

### 常见配置问题

1. #### API 密钥无效

   ```bash
   # 检查API密钥配置
   echo $ARK_API_KEY
   curl -H "Authorization: Bearer $ARK_API_KEY" $ARK_BASE_URL/models
   ```text

2. #### 数据库连接失败

   ```bash
   # 测试数据库连接
   psql $DATABASE_URL -c "SELECT 1;"
   ```text

3. #### Redis 连接失败

   ```bash
   # 测试Redis连接
   redis-cli -u $REDIS_URL ping
   ```text

4. #### 服务间通信失败

   ```bash
   # 检查服务发现
   nslookup algo-service
   curl http://algo-service:8000/health
   ```text

### 配置验证脚本

```bash
#!/bin/bash
# config-validator.sh

echo "🔍 验证环境配置..."

# 检查必需变量

required_vars=("ARK_API_KEY" "DATABASE_URL" "REDIS_URL")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "❌ 缺少必需环境变量: $var"
        exit 1
    else
        echo "✅ $var 已配置"
    fi
done

# 测试数据库连接

if psql $DATABASE_URL -c "SELECT 1;" > /dev/null 2>&1; then
    echo "✅ 数据库连接正常"
else
    echo "❌ 数据库连接失败"
    exit 1
fi

# 测试Redis连接

if redis-cli -u $REDIS_URL ping > /dev/null 2>&1; then
    echo "✅ Redis连接正常"
else
    echo "❌ Redis连接失败"
    exit 1
fi

echo "🎉 所有配置验证通过！"
```text

## 📚 相关文档

- [统一部署指南](UNIFIED_DEPLOYMENT_GUIDE.md)

- [故障排除指南](TROUBLESHOOTING_GUIDE.md)
- [安全最佳实践](BEST_PRACTICES.md#安全最佳实践)

- [性能优化指南](BEST_PRACTICES.md#性能优化最佳实践)

---

#### 配置完成！ 🎉

如有问题，请参考 [故障排除指南](TROUBLESHOOTING_GUIDE.md) 或提交 Issue。
