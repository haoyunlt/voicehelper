# VoiceHelper AI - Docker Compose 部署指南

## 🚀 快速开始

### 一键启动
```bash
# 使用快速启动脚本（推荐）
./quick-start.sh

# 或使用 Makefile
make quick-start
```

### 手动启动
```bash
# 1. 复制环境配置
cp env.unified .env

# 2. 启动开发环境
./deploy.sh -e dev up -d

# 3. 查看状态
./deploy.sh status
```

## 📋 系统要求

### 最低要求
- **操作系统**: Linux, macOS, Windows (WSL2)
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **内存**: 4GB 可用内存
- **存储**: 10GB 可用空间
- **网络**: 互联网连接（用于拉取镜像和 API 调用）

### 推荐配置
- **内存**: 8GB+ 可用内存
- **CPU**: 4核心+
- **存储**: 20GB+ 可用空间（包含日志和数据）

## 🏗️ 部署架构

### 服务组件
```
┌─────────────────────────────────────────────────────────────┐
│                    VoiceHelper AI 架构                      │
├─────────────────────────────────────────────────────────────┤
│  🌐 负载均衡层                                               │
│  ├── Nginx (反向代理)                                       │
│  └── HAProxy (生产环境负载均衡)                              │
├─────────────────────────────────────────────────────────────┤
│  🖥️ 应用服务层                                               │
│  ├── Next.js Frontend (Web 界面)                           │
│  ├── Go Gateway (API 网关)                                 │
│  ├── Python Algo Service (AI 算法服务)                     │
│  ├── Python Voice Service (语音处理)                       │
│  ├── Developer Portal (开发者门户)                          │
│  └── Admin Panel (管理后台)                                │
├─────────────────────────────────────────────────────────────┤
│  🗄️ 数据存储层                                               │
│  ├── PostgreSQL (主数据库)                                 │
│  ├── Redis (缓存和会话)                                     │
│  ├── Milvus (向量数据库)                                    │
│  ├── Neo4j (图数据库)                                       │
│  └── NATS (消息队列)                                        │
├─────────────────────────────────────────────────────────────┤
│  📊 监控和工具层                                             │
│  ├── Prometheus (指标收集)                                  │
│  ├── Grafana (数据可视化)                                   │
│  ├── Jaeger (链路追踪)                                      │
│  ├── pgAdmin (数据库管理)                                   │
│  ├── Redis Commander (Redis 管理)                          │
│  └── Attu (Milvus 管理)                                    │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 部署模式

### 1. 开发模式 (dev)
适用于本地开发和调试：
```bash
# 启动开发环境
./deploy.sh -e dev up -d

# 或使用 Makefile
make dev
```

**特性**:
- 热重载支持
- 详细日志输出
- 开发工具集成
- 代码挂载卷

### 2. 生产模式 (prod)
适用于生产环境部署：
```bash
# 启动生产环境
./deploy.sh -e prod up -d

# 或使用 Makefile
make prod
```

**特性**:
- 性能优化配置
- 多副本部署
- 资源限制
- 安全加固
- 监控和告警

### 3. 本地模式 (local)
适用于快速体验：
```bash
# 启动本地环境
./deploy.sh -e local up -d

# 或使用现有配置
make local
```

**特性**:
- 简化配置
- 快速启动
- 基础功能

## 🎯 服务配置

### 核心服务 (core)
仅启动必要的核心服务：
```bash
./deploy.sh -p core up -d
# 或
make core
```

包含：
- PostgreSQL, Redis, Milvus, NATS
- Gateway, Algo Service, Voice Service

### 监控服务 (monitoring)
仅启动监控相关服务：
```bash
./deploy.sh -p monitoring up -d
# 或
make monitoring
```

包含：
- Prometheus, Grafana, Jaeger

### 开发工具 (tools)
仅启动开发和管理工具：
```bash
./deploy.sh -p tools up -d
# 或
make tools
```

包含：
- pgAdmin, Redis Commander, Attu, Mailhog, Swagger UI

## 🔑 环境配置

### API 密钥配置
编辑 `.env` 文件，配置以下 API 密钥：

```bash
# 豆包大模型 (推荐)
ARK_API_KEY=your-ark-api-key-here

# GLM-4 (备用)
GLM_API_KEY=your-glm-api-key-here

# Azure 语音服务
AZURE_SPEECH_KEY=your-azure-speech-key-here
AZURE_SPEECH_REGION=eastus
```

### 获取 API 密钥
1. **豆包 API**: https://console.volcengine.com/
2. **GLM-4 API**: https://open.bigmodel.cn/
3. **Azure 语音**: https://portal.azure.com/

### 端口配置
默认端口映射：
```
Web 应用:      3000
API 网关:      8080
算法服务:      8000
语音服务:      8001
管理后台:      5001
PostgreSQL:    5432
Redis:         6379
Milvus:        19530
Neo4j HTTP:    7474
Neo4j Bolt:    7687
NATS:          4222
```

## 📊 访问地址

### 开发环境
- **Web 应用**: http://localhost:3000
- **API 网关**: http://localhost:8080
- **算法服务**: http://localhost:8000
- **语音服务**: http://localhost:8001
- **管理后台**: http://localhost:5001

### 管理工具
- **Grafana**: http://localhost:3004 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **pgAdmin**: http://localhost:5050 (admin@voicehelper.ai/admin123)
- **Redis 管理**: http://localhost:8081
- **Milvus 管理**: http://localhost:3001
- **API 文档**: http://localhost:8082

### 生产环境
- **Web 应用**: http://localhost:80
- **HAProxy 统计**: http://localhost:8404/stats

## 🛠️ 常用命令

### 基础操作
```bash
# 查看服务状态
./deploy.sh status
make status

# 查看日志
./deploy.sh logs
make logs

# 重启服务
./deploy.sh restart
make restart

# 停止服务
./deploy.sh down
make stop

# 健康检查
./deploy.sh health
make health
```

### 镜像管理
```bash
# 构建镜像
./deploy.sh build
make build

# 拉取最新镜像
./deploy.sh pull
make pull

# 清理未使用资源
./deploy.sh -f clean
make clean
```

### 数据管理
```bash
# 备份数据
./deploy.sh backup
make backup

# 恢复数据
./deploy.sh restore /path/to/backup
make restore BACKUP_DIR=/path/to/backup
```

### 扩缩容
```bash
# 扩容算法服务到 3 个实例
./deploy.sh scale algo-service=3
make scale-algo REPLICAS=3

# 扩容网关服务到 2 个实例
./deploy.sh scale gateway=2
make scale-gateway REPLICAS=2
```

## 🔍 故障排除

### 常见问题

#### 1. 端口冲突
**错误**: `Port already in use`
**解决**:
```bash
# 查看端口占用
lsof -i :3000
netstat -tulpn | grep :3000

# 停止冲突服务或修改端口配置
```

#### 2. 内存不足
**错误**: `Cannot allocate memory`
**解决**:
```bash
# 检查内存使用
free -h
docker system df

# 清理未使用资源
docker system prune -f
```

#### 3. 磁盘空间不足
**错误**: `No space left on device`
**解决**:
```bash
# 检查磁盘使用
df -h
docker system df

# 清理 Docker 资源
docker system prune -a -f
docker volume prune -f
```

#### 4. 服务启动失败
**排查步骤**:
```bash
# 1. 查看服务状态
./deploy.sh status

# 2. 查看详细日志
./deploy.sh logs -s <service-name>

# 3. 检查配置文件
cat .env

# 4. 重启特定服务
docker-compose restart <service-name>
```

#### 5. API 密钥配置错误
**症状**: AI 功能无法使用
**解决**:
```bash
# 检查 API 密钥配置
grep -E "(ARK_API_KEY|GLM_API_KEY)" .env

# 重新配置密钥
./quick-start.sh
```

### 日志查看
```bash
# 查看所有服务日志
./deploy.sh logs

# 查看特定服务日志
./deploy.sh -s gateway logs
./deploy.sh -s algo-service logs

# 实时跟踪日志
./deploy.sh logs -f

# 查看最近 100 行日志
docker-compose logs --tail=100 gateway
```

### 性能监控
```bash
# 查看资源使用情况
docker stats

# 查看服务健康状态
./deploy.sh health

# 访问 Grafana 监控面板
open http://localhost:3004
```

## 🔒 安全配置

### 生产环境安全建议

1. **修改默认密码**:
```bash
# 修改数据库密码
POSTGRES_PASSWORD=your-secure-password
REDIS_PASSWORD=your-secure-password

# 修改管理界面密码
# Grafana: admin/your-secure-password
# pgAdmin: admin@voicehelper.ai/your-secure-password
```

2. **启用 HTTPS**:
```bash
# 配置 SSL 证书
mkdir -p tools/deployment/ssl
# 将证书文件放置到 ssl 目录
```

3. **网络安全**:
```bash
# 限制外部访问端口
# 仅暴露必要的端口 (80, 443)
# 使用防火墙规则限制访问
```

4. **定期更新**:
```bash
# 更新镜像
./deploy.sh pull

# 重新部署
./deploy.sh restart
```

## 📈 性能优化

### 资源配置调优

1. **内存配置**:
```yaml
# docker-compose.prod.yml
services:
  postgres:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

2. **CPU 配置**:
```yaml
services:
  algo-service:
    deploy:
      resources:
        limits:
          cpus: '4'
        reservations:
          cpus: '2'
```

3. **存储优化**:
```bash
# 使用 SSD 存储
# 配置数据库连接池
# 启用 Redis 持久化
```

### 扩容策略

1. **水平扩容**:
```bash
# 增加服务实例
./deploy.sh scale gateway=3
./deploy.sh scale algo-service=2
```

2. **负载均衡**:
```bash
# 启用 HAProxy 负载均衡
docker-compose -f docker-compose.yml -f docker-compose.prod.yml --profile loadbalancer up -d
```

## 🔄 更新和维护

### 版本更新
```bash
# 1. 备份数据
./deploy.sh backup

# 2. 拉取最新代码
git pull origin main

# 3. 更新镜像
./deploy.sh pull

# 4. 重新部署
./deploy.sh restart

# 5. 验证更新
./deploy.sh health
```

### 定期维护
```bash
# 每周执行
./deploy.sh backup
docker system prune -f

# 每月执行
./deploy.sh -f clean
# 检查日志文件大小
# 更新依赖和镜像
```

## 📞 技术支持

### 获取帮助
- **项目文档**: [docs/](./docs/)
- **问题反馈**: [GitHub Issues](https://github.com/your-org/voicehelper/issues)
- **技术讨论**: [GitHub Discussions](https://github.com/your-org/voicehelper/discussions)

### 联系方式
- **邮箱**: support@voicehelper.ai
- **官网**: https://voicehelper.ai
- **文档**: https://docs.voicehelper.ai

---

## 📝 更新日志

### v2.0.0 (2024-12-XX)
- ✨ 完整的 Docker Compose 部署方案
- 🚀 一键启动脚本
- 📊 完整的监控和管理工具
- 🔒 生产级安全配置
- 📈 性能优化和扩容支持

### v1.9.0 (2024-11-XX)
- 🎯 核心功能完成
- 🤖 多模型 AI 支持
- 🎤 语音处理功能
- 📱 多平台支持
