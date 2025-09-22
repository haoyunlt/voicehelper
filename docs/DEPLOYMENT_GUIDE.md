# VoiceHelper 统一部署指南

## 📋 部署概述

本指南整合了所有部署相关文档，提供从开发环境到生产环境的完整部署方案。

## 🎯 部署方式对比

| 部署方式 | 适用场景 | 复杂度 | 推荐指数 | 文档位置 |
|----------|----------|--------|----------|----------|
| #### Docker Compose | 开发、测试、小规模生产 | ⭐⭐ | ⭐⭐⭐⭐⭐ | [Docker Compose 部署](#docker-compose-部署) |
| #### Kubernetes | 大规模生产、云原生 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | [Kubernetes 部署](#kubernetes-部署) |
| #### 手动部署 | 特殊环境、定制需求 | ⭐⭐⭐⭐⭐ | ⭐⭐ | [手动部署](#手动部署) |

## 🖥️ 环境要求

### 最小配置

- #### CPU: 4核心

- #### 内存: 8GB
- #### 存储: 20GB可用空间

- #### 网络: 稳定的互联网连接

### 推荐配置

- #### CPU: 8核心

- #### 内存: 16GB
- #### 存储: 50GB SSD

- #### 网络: 千兆网络

### 软件要求

- #### Docker: 20.10+

- #### Docker Compose: 2.0+
- #### Node.js: 18+ (开发环境)

- #### Python: 3.11+ (开发环境)
- #### Go: 1.21+ (开发环境)

- #### Kubernetes: 1.24+ (生产环境)
- #### Helm: 3.8+ (K8s部署)

## 🐳 Docker Compose 部署

### 快速开始

```bash
# 1. 克隆项目

git clone https://github.com/your-org/voicehelper.git
cd voicehelper

# 2. 配置环境变量

cp deploy/config/env.local .env
# 编辑 .env 文件，填入必要的API密钥

# 3. 启动所有服务

docker-compose up -d

# 4. 验证部署

curl http://localhost:8080/health
curl http://localhost:8000/health
curl http://localhost:3000
```text

### 服务访问地址

| 服务 | 地址 | 说明 |
|------|------|------|
| #### Web界面 | http://localhost:3000 | Next.js前端应用 |
| #### API网关 | http://localhost:8080 | Go后端服务 |
| #### 算法服务 | http://localhost:8000 | Python算法服务 |
| #### API文档 | http://localhost:8000/docs | FastAPI Swagger |
| #### 监控面板 | http://localhost:3001 | Grafana仪表板 |
| #### 数据库 | localhost:5432 | PostgreSQL |
| #### 缓存 | localhost:6379 | Redis |
| #### 向量库 | localhost:19530 | Milvus |

### 环境变量配置

#### 必需配置

```bash
# 豆包API配置（必须）

ARK_API_KEY=your_ark_api_key_here
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL=ep-20241201140014-vbzjz

# 数据库配置

DATABASE_URL=postgres://chatbot:chatbot123@postgres:5432/chatbot?sslmode=disable
REDIS_URL=redis://redis:6379
```text

#### 可选配置

```bash
# 日志级别

LOG_LEVEL=info

# JWT配置

JWT_SECRET=your_jwt_secret_here
ADMIN_SECRET_KEY=your_admin_secret_here

# 性能优化

WORKER_PROCESSES=4
MAX_CONNECTIONS=100
```text

## ☸️ Kubernetes 部署

### 生产环境部署

```bash
# 1. 创建命名空间

kubectl create namespace voicehelper

# 2. 部署配置

kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/secrets.yaml

# 3. 部署服务

kubectl apply -f deploy/k8s/postgres.yaml
kubectl apply -f deploy/k8s/redis.yaml
kubectl apply -f deploy/k8s/milvus.yaml
kubectl apply -f deploy/k8s/backend.yaml
kubectl apply -f deploy/k8s/algo.yaml
kubectl apply -f deploy/k8s/frontend.yaml

# 4. 部署监控

kubectl apply -f deploy/k8s/monitoring.yaml

# 5. 部署入口

kubectl apply -f deploy/k8s/ingress.yaml
```text

### 高可用配置

```yaml
# 示例：后端服务高可用配置

apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    spec:
      containers:
      - name: backend

        image: voicehelper/backend:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```text

## 🔧 手动部署

### 后端服务部署

```bash
# 1. 构建Go服务

cd backend
go mod download
go build -o server cmd/server/main.go

# 2. 配置环境变量

export DATABASE_URL="postgres://chatbot:chatbot123@localhost:5432/chatbot?sslmode=disable"
export REDIS_URL="redis://localhost:6379"
export ALGO_SERVICE_URL="http://localhost:8000"

# 3. 启动服务

./server
```text

### 算法服务部署

```bash
# 1. 创建虚拟环境

cd algo
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖

pip install -r requirements.txt

# 3. 配置环境变量

export MILVUS_HOST="localhost"
export MILVUS_PORT="19530"
export ARK_API_KEY="your_api_key"

# 4. 启动服务

python -m uvicorn main:app --host 0.0.0.0 --port 8000
```text

### 前端服务部署

```bash
# 1. 安装依赖

cd frontend
npm install

# 2. 构建生产版本

npm run build

# 3. 启动服务

npm start
```text

## 📊 监控和日志

### 系统监控

```bash
# 检查服务状态

docker-compose ps

# 查看日志

docker-compose logs -f backend
docker-compose logs -f algo
docker-compose logs -f frontend

# 监控资源使用

docker stats
```text

### 性能监控

```bash
# 运行性能测试

python tests/performance/unified_performance_test.py

# 运行基准测试

python tests/unified_benchmark_test.py

# 查看监控面板

open http://localhost:3001  # Grafana
```text

## 🔒 安全配置

### 生产环境安全清单

- [ ] 修改默认密码

- [ ] 配置HTTPS证书
- [ ] 设置防火墙规则

- [ ] 启用访问日志
- [ ] 配置备份策略

- [ ] 设置监控告警

### 网络安全

```bash
# 防火墙配置示例

ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 3000/tcp  # 前端
ufw allow 8080/tcp  # 后端API
ufw enable
```text

## 🚨 故障排除

### 常见问题

1. #### 服务启动失败

   ```bash
   # 检查端口占用
   netstat -tulpn | grep :8080

   # 检查日志
   docker-compose logs backend
   ```text

2. #### 数据库连接失败

   ```bash
   # 检查数据库状态
   docker-compose exec postgres psql -U chatbot -d chatbot -c "SELECT 1;"
   ```text

3. #### API响应慢

   ```bash
   # 运行性能诊断
   python tests/performance/unified_performance_test.py --test-type quick
   ```text

### 性能优化

```bash
# 数据库优化

docker-compose exec postgres psql -U chatbot -d chatbot -c "
CREATE INDEX CONCURRENTLY idx_messages_created_at ON messages(created_at);
CREATE INDEX CONCURRENTLY idx_documents_content ON documents USING gin(to_tsvector('english', content));
"

# Redis优化

docker-compose exec redis redis-cli CONFIG SET maxmemory 1gb
docker-compose exec redis redis-cli CONFIG SET maxmemory-policy allkeys-lru
```text

## 📈 扩展部署

### 水平扩展

```yaml
# 后端服务扩展

apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
spec:
  replicas: 5  # 扩展到5个实例
  template:
    spec:
      containers:
      - name: backend

        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "200m"
```text

### 数据库扩展

```bash
# PostgreSQL读写分离

# 主库：写入操作
# 从库：读取操作

# 连接池：PgBouncer
```text

## 🔄 备份和恢复

### 数据备份

```bash
# 数据库备份

docker-compose exec postgres pg_dump -U chatbot chatbot > backup_$(date +%Y%m%d).sql

# Redis备份

docker-compose exec redis redis-cli BGSAVE
docker cp $(docker-compose ps -q redis):/data/dump.rdb ./redis_backup_$(date +%Y%m%d).rdb

# 文件备份

tar -czf voicehelper_backup_$(date +%Y%m%d).tar.gz uploads/ logs/
```text

### 灾难恢复

```bash
# 恢复数据库

docker-compose exec postgres psql -U chatbot -d chatbot < backup_20241201.sql

# 恢复Redis

docker cp redis_backup_20241201.rdb $(docker-compose ps -q redis):/data/dump.rdb
docker-compose restart redis
```text

## 📚 相关文档

- [环境配置指南](ENVIRONMENT_CONFIG.md)

- [故障排除指南](TROUBLESHOOTING_GUIDE.md)
- [性能优化指南](BEST_PRACTICES.md#性能优化)

- [安全最佳实践](BEST_PRACTICES.md#安全最佳实践)

---

#### 部署完成！ 🎉

如有问题，请参考 [故障排除指南](TROUBLESHOOTING_GUIDE.md) 或提交 Issue。
