# 🚀 本地部署指南

智能聊天机器人系统完整本地部署方案，包含所有第三方软件的自动化部署。

## 📋 系统要求

### 硬件要求
- **CPU**: 4核心以上
- **内存**: 8GB以上（推荐16GB）
- **存储**: 20GB可用空间
- **网络**: 稳定的互联网连接

### 软件要求
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **操作系统**: macOS/Linux/Windows (WSL2)

## 🎯 快速开始

### 方式一：智能部署（推荐）

```bash
# 克隆项目
git clone <repository-url>
cd chatbot

# 智能检测并部署（自动检测缺失的服务）
./deploy.sh

# 或指定部署模式
./deploy.sh --full      # 完整部署所有服务
./deploy.sh --chatbot   # 仅部署应用服务
./deploy.sh --infra     # 仅部署基础设施
```

### 方式二：快速环境启动

```bash
# 开发环境（最小化）
./deploy-helper.sh quick dev

# 测试环境（完整）
./deploy-helper.sh quick test

# 演示环境（含监控）
./deploy-helper.sh quick demo
```

> 注：部署脚本实际位于 `deploy/scripts/` 目录，根目录的脚本为便捷入口

### 方式三：使用 Makefile

```bash
# 完整部署
make -f deploy/Makefile.local setup

# 或者分步骤
make -f deploy/Makefile.local check-deps  # 检查依赖
make -f deploy/Makefile.local build       # 构建镜像
make -f deploy/Makefile.local up          # 启动服务
```

## 🏗️ 架构概览

### 服务组件

| 服务类型 | 服务名称 | 端口 | 说明 |
|---------|---------|------|------|
| **应用服务** | Frontend | 3000 | Next.js前端 |
| | Gateway | 8080 | Go后端网关 |
| | Algorithm | 8000 | Python算法服务 |
| | Admin | 5001 | 管理后台 |
| **数据存储** | PostgreSQL | 5432 | 关系数据库 |
| | Redis | 6379 | 缓存数据库 |
| | Milvus | 19530 | 向量数据库 |
| | Neo4j | 7474/7687 | 图数据库 |
| **监控服务** | Prometheus | 9090 | 指标收集 |
| | Grafana | 3001 | 监控面板 |
| | cAdvisor | 8081 | 容器监控 |
| **日志服务** | Elasticsearch | 9200 | 日志存储 |
| | Kibana | 5601 | 日志分析 |
| **消息队列** | RabbitMQ | 5672/15672 | 消息中间件 |
| **文件存储** | MinIO | 9000/9001 | 对象存储 |

### 网络架构

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Network                       │
│                 (172.20.0.0/16)                        │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  Frontend   │  │   Gateway   │  │  Algorithm  │     │
│  │   :3000     │  │    :8080    │  │    :8000    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│         │                 │                 │           │
│  ┌──────┴─────────────────┴─────────────────┴──────┐    │
│  │              Data Layer                         │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐│    │
│  │  │PostgreSQL│ │  Redis  │ │ Milvus  │ │ Neo4j   ││    │
│  │  │  :5432   │ │  :6379  │ │ :19530  │ │ :7474   ││    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘│    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Monitoring Layer                   │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐│    │
│  │  │Prometheus│ │ Grafana │ │Elasticsearch│ │Kibana││    │
│  │  │  :9090   │ │  :3001  │ │  :9200  │ │ :5601   ││    │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘│    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## ⚙️ 配置说明

### 环境变量配置

复制并编辑环境变量文件：

```bash
cp env.local .env
```

**重要配置项**：

```bash
# LLM API Keys (必须配置)
OPENAI_API_KEY=your-openai-api-key-here
ARK_API_KEY=your-ark-api-key-here

# 微信小程序 (可选)
WECHAT_APP_ID=your-wechat-app-id
WECHAT_APP_SECRET=your-wechat-app-secret

# JWT密钥 (生产环境必须修改)
JWT_SECRET=your-jwt-secret-key-for-development
```

### 数据库配置

系统会自动创建以下数据库：

- **PostgreSQL**: `chatbot` 数据库，用户 `chatbot/chatbot123`
- **Redis**: 密码 `redis123`
- **Neo4j**: 用户 `neo4j/neo4j123`

## 🚀 部署模式说明

### 智能部署模式

`deploy.sh` 支持多种部署模式：

| 模式 | 命令 | 说明 |
|------|------|------|
| **智能模式** | `./deploy.sh` | 自动检测环境，仅部署缺失的服务 |
| **完整部署** | `./deploy.sh --full` | 部署所有基础设施和应用服务 |
| **应用部署** | `./deploy.sh --chatbot` | 仅部署聊天机器人应用（自动检查依赖） |
| **基础设施** | `./deploy.sh --infra` | 仅部署基础设施服务 |
| **单服务** | `./deploy.sh --service redis` | 部署特定服务 |

### 部署选项

```bash
# 强制重新部署（即使服务已存在）
./deploy.sh --full --force

# 跳过镜像构建
./deploy.sh --chatbot --skip-build

# 显示详细日志
./deploy.sh --full --verbose

# 查看服务状态
./deploy.sh --status

# 清理所有服务
./deploy.sh --clean
```

## 🔧 常用命令

### 基础操作

```bash
# 查看服务状态
make -f deploy/Makefile.local status

# 查看日志
make -f deploy/Makefile.local logs

# 实时查看日志
make -f deploy/Makefile.local logs-f

# 查看特定服务日志
make -f deploy/Makefile.local logs-postgres
make -f deploy/Makefile.local logs-gateway
```

### 服务管理

```bash
# 启动所有服务
make -f deploy/Makefile.local up

# 停止所有服务
make -f deploy/Makefile.local down

# 重启服务
make -f deploy/Makefile.local restart

# 仅启动基础设施（开发模式）
make -f deploy/Makefile.local dev
```

### 数据库管理

```bash
# 初始化数据库
make -f deploy/Makefile.local db-init

# 重置数据库
make -f deploy/Makefile.local db-reset

# 备份数据库
make -f deploy/Makefile.local db-backup
```

### 监控和调试

```bash
# 健康检查
make -f deploy/Makefile.local health

# 查看资源使用
make -f deploy/Makefile.local stats

# 进入容器
make -f deploy/Makefile.local shell-postgres
make -f deploy/Makefile.local shell-redis
```

## 🌐 访问地址

### 应用服务
- **前端界面**: http://localhost:3000
- **API网关**: http://localhost:8080
- **算法服务**: http://localhost:8000/docs (Swagger)
- **管理后台**: http://localhost:5001

### 监控面板
- **Grafana**: http://localhost:3001 (`admin/admin123`)
- **Prometheus**: http://localhost:9090
- **Kibana**: http://localhost:5601

### 数据库管理
- **Neo4j Browser**: http://localhost:7474 (`neo4j/neo4j123`)
- **RabbitMQ**: http://localhost:15672 (`rabbitmq/rabbitmq123`)
- **MinIO Console**: http://localhost:9001 (`minioadmin/minioadmin`)

## 🛠️ 高级功能

### 部署助手工具

`deploy-helper.sh` 提供交互式管理界面：

```bash
# 打开交互式菜单
./deploy-helper.sh

# 或直接执行功能
./deploy-helper.sh health      # 健康检查报告
./deploy-helper.sh backup      # 备份数据
./deploy-helper.sh diagnose    # 故障诊断
./deploy-helper.sh monitor     # 打开监控面板
```

### 数据备份与恢复

```bash
# 自动备份所有数据
./deploy-helper.sh backup

# 恢复数据
./deploy-helper.sh restore backups/20240321_143022.tar.gz
```

### 环境检测功能

系统会自动检测：
- ✅ 已部署的服务（跳过）
- ⚠️ 未运行的服务（重启）
- ❌ 缺失的服务（自动部署）
- 🔍 端口冲突（提示解决）

## 🔍 故障排查

### 常见问题

#### 1. 端口冲突
```bash
# 检查端口占用
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis
lsof -i :19530 # Milvus

# 解决方案：停止占用端口的服务或修改配置
```

#### 2. 内存不足
```bash
# 查看内存使用
docker stats

# 解决方案：
# - 增加系统内存
# - 调整服务配置
# - 分批启动服务
```

#### 3. 服务启动失败
```bash
# 查看具体错误
docker-compose -f docker-compose.local.yml logs [service-name]

# 常见解决方案：
# - 检查配置文件
# - 确认依赖服务已启动
# - 检查磁盘空间
```

#### 4. 网络连接问题
```bash
# 检查网络
docker network ls
docker network inspect chatbot_chatbot-network

# 重建网络
make -f Makefile.local clean
make -f Makefile.local up
```

### 日志位置

```bash
# 容器日志
docker-compose -f deploy/docker-compose.local.yml logs [service]

# 应用日志（如果挂载）
./deploy/local/logs/

# 数据库日志
docker exec chatbot-postgres tail -f /var/log/postgresql/postgresql.log
```

## 🧹 清理和维护

### 日常维护

```bash
# 清理未使用的镜像
docker image prune -f

# 清理未使用的卷
docker volume prune -f

# 更新基础镜像
make -f Makefile.local update
```

### 完全清理

```bash
# 停止并删除所有容器
make -f deploy/Makefile.local clean

# 删除所有数据（谨慎使用）
make -f deploy/Makefile.local clean-all
```

## 📊 性能优化

### 资源配置建议

```yaml
# docker-compose.local.yml 中的资源限制示例
services:
  postgres:
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```

### 监控指标

关键监控指标：
- **CPU使用率**: < 80%
- **内存使用率**: < 85%
- **磁盘使用率**: < 90%
- **网络延迟**: < 100ms

## 🔐 安全配置

### 生产环境注意事项

1. **修改默认密码**
2. **配置防火墙规则**
3. **启用SSL/TLS**
4. **定期备份数据**
5. **监控安全日志**

### 网络安全

```bash
# 仅暴露必要端口
# 在生产环境中移除不必要的端口映射
```

## 📞 技术支持

如遇到问题，请：

1. 查看本文档的故障排查部分
2. 检查 GitHub Issues
3. 查看项目文档：`docs/PROJECT_MASTER_DOC.md`

---

**祝您部署顺利！** 🎉
