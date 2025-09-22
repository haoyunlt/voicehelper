# Docker Compose 本地环境搭建指南

## 📋 概述

本指南将帮助您使用 Docker Compose 快速搭建 VoiceHelper 项目的完整本地开发环境，包括所有必需的服务和依赖。

## 🔧 系统要求

### 必需软件
- **Docker**: >= 20.10.0
- **Docker Compose**: >= 2.0.0 (或 docker-compose >= 1.29.0)
- **Git**: 用于克隆项目
- **至少 8GB RAM**: 推荐 16GB
- **至少 20GB 磁盘空间**: 用于镜像和数据存储

### 系统兼容性
- ✅ macOS (Intel/Apple Silicon)
- ✅ Linux (Ubuntu 18.04+, CentOS 7+)
- ✅ Windows 10/11 (WSL2)

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd voicehelper
```

### 2. 环境变量配置

```bash
# 复制环境变量模板
cp env.example .env

# 编辑配置文件（必须填写 ARK_API_KEY）
vim .env
```

**必须配置的环境变量**:
```bash
# 豆包 API 配置（必填）
ARK_API_KEY=your_actual_api_key_here
```

### 3. 一键启动

```bash
# 方式一：使用初始化脚本（推荐）
./deploy/scripts/setup.sh

# 方式二：使用 Makefile
cd deploy
make setup

# 方式三：手动启动
docker-compose -f deploy/docker-compose.local.yml up -d
```

### 4. 验证部署

```bash
# 检查服务状态
cd deploy
make status

# 或者直接查看
docker-compose -f docker-compose.local.yml ps
```

## 📦 服务架构

### 核心应用服务
| 服务名 | 端口 | 说明 | 健康检查 |
|--------|------|------|----------|
| **frontend** | 3000 | Next.js 前端界面 | http://localhost:3000 |
| **backend** | 8080 | Go 后端 API 服务 | http://localhost:8080/healthz |
| **algo-service** | 8000 | Python 算法服务 | http://localhost:8000/health |
| **admin** | 5001 | 管理后台 | http://localhost:5001 |

### 数据存储服务
| 服务名 | 端口 | 说明 | 管理界面 |
|--------|------|------|----------|
| **postgres** | 5432 | 主数据库 | - |
| **redis** | 6379 | 缓存数据库 | - |
| **milvus** | 19530 | 向量数据库 | http://localhost:9091 |
| **neo4j** | 7474/7687 | 图数据库 | http://localhost:7474 |

### 监控和日志服务
| 服务名 | 端口 | 说明 | 访问地址 |
|--------|------|------|----------|
| **prometheus** | 9090 | 指标收集 | http://localhost:9090 |
| **grafana** | 3001 | 监控面板 | http://localhost:3001 |
| **elasticsearch** | 9200 | 日志存储 | http://localhost:9200 |
| **kibana** | 5601 | 日志分析 | http://localhost:5601 |
| **rabbitmq** | 15672 | 消息队列管理 | http://localhost:15672 |

## 🛠️ 常用命令

### 基础操作

```bash
cd deploy

# 启动所有服务
make up

# 停止所有服务
make down

# 重启所有服务
make restart

# 查看服务状态
make status

# 查看服务日志
make logs

# 实时查看日志
make logs-f

# 查看特定服务日志
make logs-backend
make logs-algo-service
```

### 开发调试

```bash
# 仅启动基础设施（用于本地开发）
make dev

# 构建应用镜像
make build

# 健康检查
make health

# 查看资源使用情况
make stats

# 进入容器调试
make shell-postgres
make shell-redis
```

### 数据库管理

```bash
# 初始化数据库
make db-init

# 重置数据库
make db-reset

# 备份数据库
make db-backup
```

### 清理操作

```bash
# 清理容器和网络
make clean

# 完全清理（包括数据卷）⚠️
make clean-all
```

## 🔍 服务验证

### 1. 基础服务检查

```bash
# 检查所有端口是否正常监听
netstat -tlnp | grep -E "(3000|8080|8000|5432|6379|19530)"

# 或使用 nc 命令逐个检查
nc -zv localhost 5432  # PostgreSQL
nc -zv localhost 6379  # Redis  
nc -zv localhost 19530 # Milvus
```

### 2. 应用服务测试

```bash
# 测试后端 API
curl http://localhost:8080/healthz

# 测试算法服务
curl http://localhost:8000/health

# 测试前端页面
curl -I http://localhost:3000
```

### 3. 数据库连接测试

```bash
# PostgreSQL 连接测试
docker exec -it chatbot-postgres psql -U chatbot -d chatbot -c "SELECT version();"

# Redis 连接测试
docker exec -it chatbot-redis redis-cli ping

# Milvus 连接测试
curl http://localhost:9091/healthz
```

## 🎯 开发工作流

### 本地开发模式

```bash
# 1. 仅启动基础设施服务
make dev

# 2. 本地运行应用服务
cd ../backend && go run cmd/server/main.go
cd ../algo && python -m uvicorn app.main:app --reload --port 8000
cd ../frontend && npm run dev
```

### 完整容器模式

```bash
# 1. 构建并启动所有服务
make setup

# 2. 开发时重新构建特定服务
docker-compose -f docker-compose.local.yml build backend
docker-compose -f docker-compose.local.yml up -d backend
```

### 日志调试

```bash
# 查看特定服务的详细日志
docker-compose -f docker-compose.local.yml logs -f --tail=100 backend

# 查看错误日志
docker-compose -f docker-compose.local.yml logs | grep -i error

# 查看启动日志
docker-compose -f docker-compose.local.yml logs --since=10m
```

## 🔧 配置定制

### 环境变量覆盖

创建 `.env.local` 文件覆盖默认配置：

```bash
# .env.local
ARK_API_KEY=your_production_api_key
POSTGRES_PASSWORD=your_secure_password
REDIS_PASSWORD=your_redis_password

# 开发模式配置
LOG_LEVEL=debug
GIN_MODE=debug
```

### 端口映射修改

编辑 `docker-compose.local.yml` 修改端口映射：

```yaml
services:
  frontend:
    ports:
      - "3001:3000"  # 修改前端端口为 3001
  
  backend:
    ports:
      - "8081:8080"  # 修改后端端口为 8081
```

### 资源限制调整

```yaml
services:
  postgres:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

## 📊 监控面板

### Grafana 监控

1. 访问: http://localhost:3001
2. 默认账号: `admin` / `admin123`
3. 预配置的仪表板:
   - 系统资源监控
   - 应用性能监控
   - 数据库监控

### Prometheus 指标

访问 http://localhost:9090 查看原始指标数据

### 日志分析

1. Kibana: http://localhost:5601
2. 查看应用日志和错误分析
3. 设置日志告警规则

## 🐛 故障排查

### 常见问题

#### 1. 端口冲突

**症状**: 服务启动失败，提示端口被占用

**解决方案**:
```bash
# 查看端口占用
lsof -i :5432
netstat -tlnp | grep 5432

# 停止冲突服务或修改端口映射
docker-compose -f docker-compose.local.yml down
```

#### 2. 内存不足

**症状**: 容器频繁重启，OOM 错误

**解决方案**:
```bash
# 检查内存使用
docker stats

# 调整服务配置或增加系统内存
# 编辑 docker-compose.local.yml 添加内存限制
```

#### 3. 数据库连接失败

**症状**: 应用无法连接数据库

**解决方案**:
```bash
# 检查数据库容器状态
docker-compose -f docker-compose.local.yml ps postgres

# 查看数据库日志
docker-compose -f docker-compose.local.yml logs postgres

# 重启数据库服务
docker-compose -f docker-compose.local.yml restart postgres
```

#### 4. Milvus 启动失败

**症状**: 向量数据库无法启动

**解决方案**:
```bash
# 检查依赖服务
docker-compose -f docker-compose.local.yml ps etcd minio

# 清理并重启
docker-compose -f docker-compose.local.yml down
docker volume rm $(docker volume ls -q | grep milvus)
docker-compose -f docker-compose.local.yml up -d
```

#### 5. API 密钥错误

**症状**: AI 功能无法使用，API 调用失败

**解决方案**:
```bash
# 检查环境变量
docker-compose -f docker-compose.local.yml exec algo-service env | grep ARK

# 更新 .env 文件并重启服务
vim .env
docker-compose -f docker-compose.local.yml restart algo-service
```

### 诊断命令

```bash
# 完整健康检查
make health

# 查看所有容器状态
docker ps -a

# 查看网络连接
docker network ls
docker network inspect chatbot-network

# 查看数据卷
docker volume ls | grep chatbot

# 系统资源使用
docker system df
```

### 日志收集

```bash
# 收集所有服务日志
mkdir -p logs
docker-compose -f docker-compose.local.yml logs > logs/all-services.log

# 收集系统信息
docker version > logs/docker-info.txt
docker-compose version >> logs/docker-info.txt
docker system info >> logs/docker-info.txt
```

## 🔄 更新和维护

### 更新镜像

```bash
# 更新基础镜像
make update

# 重新构建应用镜像
make rebuild
```

### 数据备份

```bash
# 备份数据库
make db-backup

# 备份配置文件
tar -czf config-backup.tar.gz .env deploy/config/
```

### 版本升级

```bash
# 1. 备份当前环境
make db-backup

# 2. 停止服务
make down

# 3. 更新代码
git pull origin main

# 4. 重新部署
make setup
```

## 📚 相关文档

- [环境变量配置指南](./ENVIRONMENT_CONFIG.md)
- [开发指南](./DEVELOPMENT_GUIDE.md)
- [API 文档](./api/openapi_v3.yaml)
- [端口使用说明](./PORT_USAGE.md)

## 🆘 获取帮助

### 社区支持
- GitHub Issues: 报告 Bug 和功能请求
- 技术文档: 查看详细的技术文档

### 快速联系
如果遇到紧急问题，请提供以下信息：
1. 操作系统和版本
2. Docker 版本信息
3. 错误日志内容
4. 复现步骤

---

**最后更新**: 2024-12-21  
**维护者**: VoiceHelper Team
