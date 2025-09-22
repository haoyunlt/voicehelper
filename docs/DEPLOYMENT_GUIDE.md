# VoiceHelper 部署指南

## 📋 目录

- [部署概述](#部署概述)
- [环境要求](#环境要求)
- [Docker Compose 部署](#docker-compose-部署)
- [Kubernetes 部署](#kubernetes-部署)
- [生产环境部署](#生产环境部署)
- [监控和日志](#监控和日志)
- [备份和恢复](#备份和恢复)
- [安全配置](#安全配置)
- [性能优化](#性能优化)
- [故障排除](#故障排除)

## 🎯 部署概述

VoiceHelper 支持多种部署方式，适应不同的环境需求：

| 部署方式 | 适用场景 | 复杂度 | 推荐指数 |
|----------|----------|--------|----------|
| Docker Compose | 开发、测试、小规模生产 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Kubernetes | 大规模生产、云原生 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 手动部署 | 特殊环境、定制需求 | ⭐⭐⭐⭐⭐ | ⭐⭐ |

### 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   Web Gateway   │    │   API Gateway   │
│   Nginx/HAProxy │◄──►│   Frontend      │◄──►│   Backend       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       ▼
         │                       │              ┌─────────────────┐
         │                       │              │   Algorithm     │
         │                       │              │   Service       │
         │                       │              └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Static Files  │    │   Database      │    │   Vector DB     │
│   CDN/Storage   │    │   PostgreSQL    │    │   Milvus        │
└─────────────────┘    │   Redis         │    │   Knowledge     │
                       └─────────────────┘    └─────────────────┘
```

## 🖥️ 环境要求

### 最小配置

| 组件 | CPU | 内存 | 存储 | 网络 |
|------|-----|------|------|------|
| **开发环境** | 4核 | 8GB | 50GB | 100Mbps |
| **测试环境** | 8核 | 16GB | 100GB | 1Gbps |
| **生产环境** | 16核 | 32GB | 500GB | 10Gbps |

### 推荐配置

| 组件 | CPU | 内存 | 存储 | 说明 |
|------|-----|------|------|------|
| **前端服务** | 2核 | 4GB | 20GB | Nginx + Next.js |
| **后端服务** | 4核 | 8GB | 50GB | Go API 服务 |
| **算法服务** | 8核 | 16GB | 100GB | Python AI 服务 |
| **数据库** | 4核 | 16GB | 200GB | PostgreSQL + Redis |
| **向量数据库** | 8核 | 32GB | 500GB | Milvus + 存储 |

### 软件依赖

**基础环境**:
- **操作系统**: Ubuntu 20.04+ / CentOS 8+ / macOS 12+
- **Docker**: >= 20.10.0
- **Docker Compose**: >= 2.0.0
- **Git**: >= 2.25.0

**可选组件**:
- **Kubernetes**: >= 1.20 (K8s 部署)
- **Helm**: >= 3.0 (K8s 包管理)
- **Nginx**: >= 1.18 (反向代理)
- **Certbot**: >= 1.0 (SSL 证书)

## 🐳 Docker Compose 部署

### 快速部署

**1. 获取代码**
```bash
# 克隆项目
git clone https://github.com/your-org/voicehelper.git
cd voicehelper

# 切换到最新稳定版本
git checkout v1.20.0
```

**2. 配置环境**
```bash
# 复制配置文件
cp deploy/config/env.local.example deploy/config/env.local

# 编辑配置文件
vim deploy/config/env.local
```

**关键配置项**:
```bash
# 基础配置
ENVIRONMENT=production
DOMAIN=your-domain.com
SSL_EMAIL=admin@your-domain.com

# 数据库配置
POSTGRES_PASSWORD=your-strong-password
REDIS_PASSWORD=your-redis-password

# AI 服务配置
OPENAI_API_KEY=sk-your-openai-key
OPENAI_BASE_URL=https://api.openai.com/v1

# 安全配置
JWT_SECRET=your-jwt-secret-key
API_KEY=your-api-key
```

**3. 启动服务**
```bash
cd deploy

# 拉取最新镜像
docker-compose pull

# 启动所有服务
docker-compose up -d

# 查看启动状态
docker-compose ps
```

**4. 验证部署**
```bash
# 等待服务启动完成
sleep 60

# 检查服务健康状态
curl -f http://localhost:8080/health
curl -f http://localhost:8000/health

# 检查前端访问
curl -f http://localhost:3000
```

### 生产环境配置

**docker-compose.prod.yml**:
```yaml
version: '3.8'

services:
  frontend:
    image: voicehelper/frontend:latest
    restart: always
    environment:
      - NODE_ENV=production
      - NEXT_PUBLIC_API_URL=https://api.your-domain.com
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.frontend.rule=Host(`your-domain.com`)"
      - "traefik.http.routers.frontend.tls.certresolver=letsencrypt"
    networks:
      - voicehelper-network

  backend:
    image: voicehelper/backend:latest
    restart: always
    environment:
      - GIN_MODE=release
      - DATABASE_URL=${DATABASE_URL}
      - REDIS_URL=${REDIS_URL}
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'
        reservations:
          memory: 1G
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - voicehelper-network

  algo-service:
    image: voicehelper/algo-service:latest
    restart: always
    environment:
      - ENVIRONMENT=production
      - MILVUS_HOST=milvus-standalone
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'
    networks:
      - voicehelper-network

  postgres:
    image: postgres:15-alpine
    restart: always
    environment:
      - POSTGRES_DB=voicehelper
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
    networks:
      - voicehelper-network

  redis:
    image: redis:7-alpine
    restart: always
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
    networks:
      - voicehelper-network

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  milvus_data:
    driver: local

networks:
  voicehelper-network:
    driver: bridge
```

### SSL 证书配置

**使用 Traefik 自动 SSL**:
```yaml
# traefik.yml
version: '3.8'

services:
  traefik:
    image: traefik:v2.10
    restart: always
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./traefik/traefik.yml:/traefik.yml:ro
      - ./traefik/acme.json:/acme.json
    networks:
      - voicehelper-network

  # 其他服务...
```

**traefik/traefik.yml**:
```yaml
api:
  dashboard: true
  insecure: true

entryPoints:
  web:
    address: ":80"
    http:
      redirections:
        entrypoint:
          to: websecure
          scheme: https
  websecure:
    address: ":443"

certificatesResolvers:
  letsencrypt:
    acme:
      email: admin@your-domain.com
      storage: acme.json
      httpChallenge:
        entryPoint: web

providers:
  docker:
    exposedByDefault: false
```

## ☸️ Kubernetes 部署

### 准备 K8s 环境

**1. 安装 Kubernetes**
```bash
# 使用 kubeadm 安装（Ubuntu）
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt update
sudo apt install -y kubelet kubeadm kubectl

# 初始化集群
sudo kubeadm init --pod-network-cidr=10.244.0.0/16

# 配置 kubectl
mkdir -p $HOME/.kube
sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
sudo chown $(id -u):$(id -g) $HOME/.kube/config

# 安装网络插件
kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
```

**2. 安装 Helm**
```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### 部署应用

**1. 创建命名空间**
```bash
kubectl create namespace voicehelper
kubectl config set-context --current --namespace=voicehelper
```

**2. 创建配置文件**
```yaml
# deploy/k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: voicehelper-config
  namespace: voicehelper
data:
  DATABASE_URL: "postgresql://postgres:password@postgres:5432/voicehelper"
  REDIS_URL: "redis://redis:6379"
  MILVUS_HOST: "milvus-standalone"
  MILVUS_PORT: "19530"
---
apiVersion: v1
kind: Secret
metadata:
  name: voicehelper-secrets
  namespace: voicehelper
type: Opaque
stringData:
  POSTGRES_PASSWORD: "your-strong-password"
  REDIS_PASSWORD: "your-redis-password"
  OPENAI_API_KEY: "sk-your-openai-key"
  JWT_SECRET: "your-jwt-secret"
```

**3. 部署数据库**
```yaml
# deploy/k8s/postgres.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: voicehelper
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: voicehelper
        - name: POSTGRES_USER
          value: postgres
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: voicehelper-secrets
              key: POSTGRES_PASSWORD
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: voicehelper
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  clusterIP: None
```

**4. 部署应用服务**
```yaml
# deploy/k8s/backend.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend
  namespace: voicehelper
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: voicehelper/backend:latest
        env:
        - name: GIN_MODE
          value: "release"
        - name: DATABASE_URL
          valueFrom:
            configMapKeyRef:
              name: voicehelper-config
              key: DATABASE_URL
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: voicehelper-config
              key: REDIS_URL
        ports:
        - containerPort: 8080
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
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1"
---
apiVersion: v1
kind: Service
metadata:
  name: backend
  namespace: voicehelper
spec:
  selector:
    app: backend
  ports:
  - port: 8080
    targetPort: 8080
  type: ClusterIP
```

**5. 部署 Ingress**
```yaml
# deploy/k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: voicehelper-ingress
  namespace: voicehelper
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - voicehelper.your-domain.com
    - api.voicehelper.your-domain.com
    secretName: voicehelper-tls
  rules:
  - host: voicehelper.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend
            port:
              number: 3000
  - host: api.voicehelper.your-domain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend
            port:
              number: 8080
```

**6. 应用配置**
```bash
# 应用所有配置
kubectl apply -f deploy/k8s/

# 检查部署状态
kubectl get pods
kubectl get services
kubectl get ingress

# 查看日志
kubectl logs -f deployment/backend
kubectl logs -f deployment/algo-service
```

### Helm Chart 部署

**1. 创建 Helm Chart**
```bash
helm create voicehelper-chart
cd voicehelper-chart
```

**2. 配置 values.yaml**
```yaml
# values.yaml
global:
  domain: voicehelper.your-domain.com
  environment: production

backend:
  image:
    repository: voicehelper/backend
    tag: latest
  replicas: 3
  resources:
    requests:
      memory: 512Mi
      cpu: 250m
    limits:
      memory: 2Gi
      cpu: 1

frontend:
  image:
    repository: voicehelper/frontend
    tag: latest
  replicas: 2
  resources:
    requests:
      memory: 256Mi
      cpu: 100m
    limits:
      memory: 1Gi
      cpu: 500m

algoService:
  image:
    repository: voicehelper/algo-service
    tag: latest
  replicas: 2
  resources:
    requests:
      memory: 1Gi
      cpu: 500m
    limits:
      memory: 4Gi
      cpu: 2

postgresql:
  enabled: true
  auth:
    postgresPassword: "your-strong-password"
  primary:
    persistence:
      size: 20Gi

redis:
  enabled: true
  auth:
    password: "your-redis-password"
  master:
    persistence:
      size: 8Gi

ingress:
  enabled: true
  className: nginx
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  tls:
    - secretName: voicehelper-tls
      hosts:
        - voicehelper.your-domain.com
```

**3. 部署 Chart**
```bash
# 安装
helm install voicehelper ./voicehelper-chart -f values.yaml

# 升级
helm upgrade voicehelper ./voicehelper-chart -f values.yaml

# 查看状态
helm status voicehelper
helm list
```

## 🏭 生产环境部署

### 高可用架构

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    │   (HAProxy/F5)  │
                    └─────────────────┘
                             │
                    ┌─────────────────┐
                    │   API Gateway   │
                    │   (Kong/Istio)  │
                    └─────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼────────┐  ┌────────▼────────┐  ┌───────▼────────┐
│   Frontend     │  │   Backend       │  │   Algorithm    │
│   (3 replicas) │  │   (5 replicas)  │  │   (3 replicas) │
└────────────────┘  └─────────────────┘  └────────────────┘
        │                    │                    │
        │           ┌────────▼────────┐           │
        │           │   Database      │           │
        │           │   (Master/Slave)│           │
        │           └─────────────────┘           │
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌─────────▼─────────┐
                    │   Shared Storage  │
                    │   (NFS/Ceph/S3)   │
                    └───────────────────┘
```

### 多环境管理

**1. 环境分离**
```bash
# 目录结构
deploy/
├── environments/
│   ├── dev/
│   │   ├── docker-compose.yml
│   │   └── .env
│   ├── staging/
│   │   ├── docker-compose.yml
│   │   └── .env
│   └── prod/
│       ├── docker-compose.yml
│       └── .env
└── scripts/
    ├── deploy-dev.sh
    ├── deploy-staging.sh
    └── deploy-prod.sh
```

**2. 部署脚本**
```bash
#!/bin/bash
# deploy/scripts/deploy-prod.sh

set -e

ENVIRONMENT="prod"
COMPOSE_FILE="environments/${ENVIRONMENT}/docker-compose.yml"
ENV_FILE="environments/${ENVIRONMENT}/.env"

echo "🚀 开始部署到 ${ENVIRONMENT} 环境..."

# 检查环境文件
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ 环境文件不存在: $ENV_FILE"
    exit 1
fi

# 拉取最新镜像
echo "📦 拉取最新镜像..."
docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE pull

# 备份数据库
echo "💾 备份数据库..."
./scripts/backup-db.sh $ENVIRONMENT

# 滚动更新
echo "🔄 执行滚动更新..."
docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d --no-deps backend
sleep 30
docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d --no-deps frontend
sleep 30
docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d --no-deps algo-service

# 健康检查
echo "🏥 执行健康检查..."
./scripts/health-check.sh $ENVIRONMENT

echo "✅ 部署完成！"
```

### 蓝绿部署

**1. 蓝绿部署脚本**
```bash
#!/bin/bash
# deploy/scripts/blue-green-deploy.sh

CURRENT_ENV=$(docker-compose ps --services | head -1 | grep -o 'blue\|green' || echo 'blue')
NEW_ENV=$([ "$CURRENT_ENV" = "blue" ] && echo "green" || echo "blue")

echo "当前环境: $CURRENT_ENV"
echo "新环境: $NEW_ENV"

# 部署到新环境
docker-compose -f docker-compose.${NEW_ENV}.yml up -d

# 等待新环境就绪
echo "等待新环境启动..."
sleep 60

# 健康检查
if ./scripts/health-check.sh $NEW_ENV; then
    echo "✅ 新环境健康检查通过"
    
    # 切换流量
    echo "🔄 切换流量到新环境..."
    ./scripts/switch-traffic.sh $NEW_ENV
    
    # 停止旧环境
    echo "🛑 停止旧环境..."
    docker-compose -f docker-compose.${CURRENT_ENV}.yml down
    
    echo "✅ 蓝绿部署完成！"
else
    echo "❌ 新环境健康检查失败，回滚..."
    docker-compose -f docker-compose.${NEW_ENV}.yml down
    exit 1
fi
```

### 数据库迁移

**1. 迁移脚本**
```bash
#!/bin/bash
# deploy/scripts/migrate-db.sh

DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-voicehelper}
DB_USER=${DB_USER:-postgres}

echo "🗄️ 开始数据库迁移..."

# 备份当前数据库
echo "💾 备份当前数据库..."
pg_dump -h $DB_HOST -p $DB_PORT -U $DB_USER $DB_NAME > backup_$(date +%Y%m%d_%H%M%S).sql

# 执行迁移
echo "🔄 执行数据库迁移..."
for migration in deploy/database/migrations/*.sql; do
    echo "执行迁移: $migration"
    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f $migration
done

echo "✅ 数据库迁移完成！"
```

**2. 迁移文件示例**
```sql
-- deploy/database/migrations/001_add_user_preferences.sql
BEGIN;

-- 添加用户偏好表
CREATE TABLE IF NOT EXISTS user_preferences (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    language VARCHAR(10) DEFAULT 'zh-CN',
    voice_enabled BOOLEAN DEFAULT true,
    theme VARCHAR(20) DEFAULT 'light',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 添加索引
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);

-- 更新版本
INSERT INTO schema_migrations (version) VALUES ('001') ON CONFLICT DO NOTHING;

COMMIT;
```

## 📊 监控和日志

### 监控系统

**1. Prometheus + Grafana**
```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources

  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'

volumes:
  prometheus_data:
  grafana_data:
```

**2. 应用监控指标**
```go
// backend/internal/metrics/metrics.go
package metrics

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

var (
    RequestsTotal = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "voicehelper_requests_total",
            Help: "Total number of requests",
        },
        []string{"method", "endpoint", "status"},
    )

    RequestDuration = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "voicehelper_request_duration_seconds",
            Help: "Request duration in seconds",
            Buckets: prometheus.DefBuckets,
        },
        []string{"method", "endpoint"},
    )

    ActiveConnections = promauto.NewGauge(
        prometheus.GaugeOpts{
            Name: "voicehelper_active_connections",
            Help: "Number of active connections",
        },
    )
)
```

### 日志管理

**1. ELK Stack 部署**
```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.8.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:8.8.0
    ports:
      - "5044:5044"
    volumes:
      - ./logging/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  kibana:
    image: docker.elastic.co/kibana/kibana:8.8.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

  filebeat:
    image: docker.elastic.co/beats/filebeat:8.8.0
    volumes:
      - ./logging/filebeat.yml:/usr/share/filebeat/filebeat.yml
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    depends_on:
      - logstash

volumes:
  elasticsearch_data:
```

**2. 日志配置**
```yaml
# logging/filebeat.yml
filebeat.inputs:
- type: container
  paths:
    - '/var/lib/docker/containers/*/*.log'
  processors:
  - add_docker_metadata:
      host: "unix:///var/run/docker.sock"

output.logstash:
  hosts: ["logstash:5044"]

logging.level: info
```

## 💾 备份和恢复

### 自动备份脚本

```bash
#!/bin/bash
# deploy/scripts/backup.sh

BACKUP_DIR="/backup/voicehelper"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=30

mkdir -p $BACKUP_DIR

echo "🗄️ 开始备份数据库..."

# 备份 PostgreSQL
docker-compose exec -T postgres pg_dump -U postgres voicehelper | gzip > $BACKUP_DIR/postgres_$DATE.sql.gz

# 备份 Redis
docker-compose exec -T redis redis-cli --rdb /data/dump.rdb
docker cp $(docker-compose ps -q redis):/data/dump.rdb $BACKUP_DIR/redis_$DATE.rdb

# 备份 Milvus 数据
docker cp $(docker-compose ps -q milvus-standalone):/var/lib/milvus $BACKUP_DIR/milvus_$DATE

# 备份配置文件
tar -czf $BACKUP_DIR/config_$DATE.tar.gz deploy/config/

# 上传到云存储（可选）
if [ ! -z "$AWS_S3_BUCKET" ]; then
    aws s3 sync $BACKUP_DIR s3://$AWS_S3_BUCKET/voicehelper-backups/
fi

# 清理旧备份
find $BACKUP_DIR -name "*.gz" -mtime +$RETENTION_DAYS -delete
find $BACKUP_DIR -name "*.rdb" -mtime +$RETENTION_DAYS -delete

echo "✅ 备份完成！"
```

### 恢复脚本

```bash
#!/bin/bash
# deploy/scripts/restore.sh

BACKUP_FILE=$1
BACKUP_DIR="/backup/voicehelper"

if [ -z "$BACKUP_FILE" ]; then
    echo "用法: $0 <backup_date>"
    echo "可用备份:"
    ls -la $BACKUP_DIR/postgres_*.sql.gz
    exit 1
fi

echo "🔄 开始恢复数据..."

# 停止服务
docker-compose down

# 恢复 PostgreSQL
echo "恢复 PostgreSQL..."
docker-compose up -d postgres
sleep 30
zcat $BACKUP_DIR/postgres_$BACKUP_FILE.sql.gz | docker-compose exec -T postgres psql -U postgres voicehelper

# 恢复 Redis
echo "恢复 Redis..."
docker-compose up -d redis
sleep 10
docker cp $BACKUP_DIR/redis_$BACKUP_FILE.rdb $(docker-compose ps -q redis):/data/dump.rdb
docker-compose restart redis

# 恢复 Milvus
echo "恢复 Milvus..."
docker cp $BACKUP_DIR/milvus_$BACKUP_FILE $(docker-compose ps -q milvus-standalone):/var/lib/milvus

# 启动所有服务
docker-compose up -d

echo "✅ 恢复完成！"
```

### 定时备份

```bash
# 添加到 crontab
crontab -e

# 每天凌晨 2 点备份
0 2 * * * /path/to/voicehelper/deploy/scripts/backup.sh >> /var/log/voicehelper-backup.log 2>&1

# 每周日凌晨 1 点清理日志
0 1 * * 0 find /var/log -name "voicehelper-*.log" -mtime +7 -delete
```

## 🔒 安全配置

### SSL/TLS 配置

**1. Let's Encrypt 证书**
```bash
# 安装 certbot
sudo apt install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d voicehelper.your-domain.com -d api.voicehelper.your-domain.com

# 自动续期
sudo crontab -e
0 12 * * * /usr/bin/certbot renew --quiet
```

**2. Nginx SSL 配置**
```nginx
# /etc/nginx/sites-available/voicehelper
server {
    listen 443 ssl http2;
    server_name voicehelper.your-domain.com;

    ssl_certificate /etc/letsencrypt/live/voicehelper.your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/voicehelper.your-domain.com/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options DENY always;
    add_header X-Content-Type-Options nosniff always;
    add_header X-XSS-Protection "1; mode=block" always;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 防火墙配置

```bash
# UFW 配置
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 允许必要端口
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS

# 内部服务端口（仅本地访问）
sudo ufw allow from 127.0.0.1 to any port 5432  # PostgreSQL
sudo ufw allow from 127.0.0.1 to any port 6379  # Redis
sudo ufw allow from 127.0.0.1 to any port 19530 # Milvus

sudo ufw --force enable
```

### 访问控制

**1. API 限流**
```nginx
# Nginx 限流配置
http {
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=1r/s;
    
    server {
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://backend;
        }
        
        location /api/v1/auth/login {
            limit_req zone=login burst=5 nodelay;
            proxy_pass http://backend;
        }
    }
}
```

**2. IP 白名单**
```bash
# 管理接口 IP 白名单
location /admin/ {
    allow 192.168.1.0/24;
    allow 10.0.0.0/8;
    deny all;
    proxy_pass http://backend;
}
```

## ⚡ 性能优化

### 数据库优化

**1. PostgreSQL 配置**
```sql
-- postgresql.conf 优化
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200

-- 创建必要索引
CREATE INDEX CONCURRENTLY idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX CONCURRENTLY idx_messages_created_at ON messages(created_at);
CREATE INDEX CONCURRENTLY idx_conversations_user_id ON conversations(user_id);
```

**2. Redis 优化**
```bash
# redis.conf 优化
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
tcp-keepalive 300
timeout 0
```

### 应用优化

**1. 连接池配置**
```go
// backend 连接池优化
db.SetMaxOpenConns(25)
db.SetMaxIdleConns(25)
db.SetConnMaxLifetime(5 * time.Minute)

// Redis 连接池
redisPool := &redis.Pool{
    MaxIdle:     10,
    MaxActive:   100,
    IdleTimeout: 240 * time.Second,
    Dial: func() (redis.Conn, error) {
        return redis.Dial("tcp", redisAddr)
    },
}
```

**2. 缓存策略**
```go
// 多级缓存
type CacheManager struct {
    l1Cache *sync.Map        // 内存缓存
    l2Cache *redis.Client    // Redis 缓存
    l3Cache *sql.DB          // 数据库
}

func (c *CacheManager) Get(key string) (interface{}, error) {
    // L1 缓存
    if val, ok := c.l1Cache.Load(key); ok {
        return val, nil
    }
    
    // L2 缓存
    val, err := c.l2Cache.Get(key).Result()
    if err == nil {
        c.l1Cache.Store(key, val)
        return val, nil
    }
    
    // L3 数据库
    val, err = c.queryDatabase(key)
    if err == nil {
        c.l2Cache.Set(key, val, time.Hour)
        c.l1Cache.Store(key, val)
    }
    
    return val, err
}
```

### CDN 配置

```bash
# 静态资源 CDN 配置
# CloudFlare / AWS CloudFront / 阿里云 CDN

# 缓存策略
Cache-Control: public, max-age=31536000  # 静态资源
Cache-Control: public, max-age=300       # API 响应
Cache-Control: no-cache                  # 动态内容
```

## 🔧 故障排除

### 常见问题诊断

**1. 服务启动失败**
```bash
# 检查服务状态
docker-compose ps
docker-compose logs service-name

# 检查资源使用
docker stats
df -h
free -h

# 检查端口占用
netstat -tlnp | grep :8080
lsof -i :8080
```

**2. 数据库连接问题**
```bash
# 检查数据库连接
docker-compose exec postgres pg_isready
docker-compose exec postgres psql -U postgres -c "SELECT version();"

# 检查连接数
docker-compose exec postgres psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# 重置连接
docker-compose restart postgres
```

**3. 性能问题诊断**
```bash
# 检查系统负载
top
htop
iostat -x 1

# 检查网络
netstat -i
ss -tuln

# 检查应用性能
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8080/health
```

### 故障恢复流程

**1. 服务异常恢复**
```bash
#!/bin/bash
# deploy/scripts/emergency-recovery.sh

echo "🚨 开始紧急恢复流程..."

# 停止所有服务
docker-compose down

# 检查磁盘空间
if [ $(df / | tail -1 | awk '{print $5}' | sed 's/%//') -gt 90 ]; then
    echo "⚠️ 磁盘空间不足，清理日志..."
    docker system prune -f
    find /var/log -name "*.log" -mtime +7 -delete
fi

# 检查内存
if [ $(free | grep Mem | awk '{print ($3/$2) * 100.0}') -gt 90 ]; then
    echo "⚠️ 内存不足，重启系统..."
    sudo reboot
fi

# 恢复服务
docker-compose up -d

# 等待服务启动
sleep 60

# 健康检查
if ./scripts/health-check.sh; then
    echo "✅ 恢复成功！"
else
    echo "❌ 恢复失败，启动备用方案..."
    ./scripts/failover.sh
fi
```

**2. 数据恢复流程**
```bash
#!/bin/bash
# deploy/scripts/data-recovery.sh

BACKUP_DATE=$1

if [ -z "$BACKUP_DATE" ]; then
    echo "用法: $0 <backup_date>"
    exit 1
fi

echo "🔄 开始数据恢复流程..."

# 创建当前数据备份
./scripts/backup.sh

# 停止服务
docker-compose down

# 恢复数据
./scripts/restore.sh $BACKUP_DATE

# 验证数据完整性
./scripts/verify-data.sh

# 启动服务
docker-compose up -d

echo "✅ 数据恢复完成！"
```

---

## 📞 技术支持

如果在部署过程中遇到问题，可以通过以下方式获取帮助：

- **文档**: [完整文档](USER_GUIDE.md)
- **GitHub Issues**: [问题反馈](https://github.com/your-org/voicehelper/issues)
- **社区讨论**: [GitHub Discussions](https://github.com/your-org/voicehelper/discussions)
- **技术支持**: support@voicehelper.com

---

**部署指南完成！** 🎉

现在你已经掌握了 VoiceHelper 的各种部署方式，可以根据实际需求选择合适的部署方案。
