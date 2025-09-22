# VoiceHelper 统一环境配置指南

## 🎯 概述

本指南说明如何使用统一的 `.env` 配置文件来管理所有服务的环境变量，实现配置的集中化管理。

## 🏗️ 配置架构

### 统一配置原则
- **单一配置源**: 所有服务共享根目录下的 `.env` 文件
- **分层覆盖**: Docker Compose 可以覆盖特定的容器内部配置
- **环境隔离**: 开发、测试、生产环境使用不同的配置文件
- **安全优先**: 敏感信息通过环境变量传递，不硬编码在代码中

### 配置文件结构
```
voicehelper/
├── .env                    # 主配置文件 (需要创建)
├── env.unified.new         # 统一配置模板 (已创建)
├── env.unified             # 旧版配置文件
├── env.example             # 配置示例文件
└── docker-compose.local.yml # Docker 编排配置
```

## 🚀 快速开始

### 1. 创建配置文件
```bash
# 复制统一配置模板
cp env.unified.new .env

# 或者从示例文件创建
cp env.example .env
```

### 2. 配置API密钥
编辑 `.env` 文件，填入你的API密钥：

```bash
# GLM-4 (智谱AI) - 推荐，成本最低
GLM_API_KEY=your-real-glm-api-key-here

# 豆包 (字节跳动) - 备选方案  
ARK_API_KEY=your-real-ark-api-key-here

# OpenAI - 可选
OPENAI_API_KEY=your-openai-api-key-here
```

### 3. 验证配置
```bash
# 运行配置验证脚本
python scripts/validate_env_config.py

# 或者直接启动服务测试
docker-compose -f docker-compose.local.yml up -d
```

## 📋 配置详解

### 🚀 基础环境配置
```bash
# 运行环境
ENV=development              # development, staging, production
NODE_ENV=development         # Next.js 环境
FLASK_ENV=development        # Flask 环境
GIN_MODE=debug              # Go Gin 模式: debug, release

# 服务名称 (自动生成容器名和日志标识)
SERVICE_NAME=voicehelper
GATEWAY_SERVICE_NAME=voicehelper-gateway
ALGO_SERVICE_NAME=voicehelper-algo
VOICE_SERVICE_NAME=voicehelper-voice
FRONTEND_SERVICE_NAME=voicehelper-frontend
ADMIN_SERVICE_NAME=voicehelper-admin
```

### 🌐 服务端口配置
```bash
# 应用服务端口
GATEWAY_PORT=8080           # API 网关
ALGO_PORT=8000              # 算法服务
VOICE_PORT=8001             # 语音服务
FRONTEND_PORT=3000          # 前端应用
PORTAL_PORT=3002            # 开发者门户
ADMIN_PORT=5001             # 管理后台

# 数据库端口
POSTGRES_PORT=5432          # PostgreSQL
REDIS_PORT=6379             # Redis
NEO4J_HTTP_PORT=7474        # Neo4j HTTP
NEO4J_BOLT_PORT=7687        # Neo4j Bolt

# 监控服务端口
PROMETHEUS_PORT=9090        # Prometheus
GRAFANA_PORT=3004           # Grafana
PGADMIN_PORT=5050           # pgAdmin
REDIS_COMMANDER_PORT=8081   # Redis Commander
```

### 🗄️ 数据库配置
```bash
# PostgreSQL 配置
POSTGRES_HOST=postgres
POSTGRES_DB=voicehelper
POSTGRES_USER=voicehelper
POSTGRES_PASSWORD=voicehelper123
DATABASE_URL=postgresql://voicehelper:voicehelper123@postgres:5432/voicehelper

# Redis 配置
REDIS_HOST=redis
REDIS_PASSWORD=redis123
REDIS_URL=redis://:redis123@redis:6379/0

# Neo4j 配置 (图数据库)
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j123
```

### 🤖 AI 模型配置
```bash
# 主要使用的模型
PRIMARY_MODEL=glm-4-flash

# GLM-4 (智谱AI) - 推荐首选
GLM_API_KEY=fc37bd957e5c4e669c748219881161b2.vnvJq6vsQIKZaNS9
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# 豆包大模型 (字节跳动) - 备选方案
ARK_API_KEY=1a208824-2b22-4a7f-ac89-49c4b1dcc5a7
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL=doubao-pro-4k

# 其他模型配置...
```

### 🔐 安全配置
```bash
# JWT 配置
JWT_SECRET=your-jwt-secret-key-change-in-production-environment
JWT_ALGORITHM=HS256
JWT_EXPIRE_HOURS=24
JWT_EXPIRES_IN=24h

# 管理员配置
ADMIN_SECRET_KEY=b29d40c1ea82b3331a76b7479526e67fcbf7e5d09719ddf8e6c607dae81756fa

# CORS 配置
CORS_ORIGINS=http://localhost:3000,http://localhost:3002,http://localhost:5001
```

## 🔄 服务配置映射

### 各服务如何读取配置

#### 1. Gateway Service (Go)
```go
// backend/cmd/server/main.go
func loadConfig() *Config {
    return &Config{
        Port:        getEnv("PORT", getEnv("GATEWAY_PORT", "8080")),
        ServiceName: getEnv("SERVICE_NAME", getEnv("GATEWAY_SERVICE_NAME", "voicehelper-gateway")),
        // ...
    }
}
```

#### 2. Algorithm Service (Python)
```python
# algo/app/main.py
SERVICE_NAME = os.getenv("SERVICE_NAME", os.getenv("ALGO_SERVICE_NAME", "voicehelper-algo"))
PORT = int(os.getenv("PORT", os.getenv("ALGO_PORT", 8000)))
```

#### 3. Admin Service (Flask)
```python
# admin/app.py
SERVICE_NAME = os.getenv('SERVICE_NAME', os.getenv('ADMIN_SERVICE_NAME', 'voicehelper-admin'))
PORT = int(os.getenv('PORT', os.getenv('ADMIN_PORT', 5001)))
```

#### 4. Frontend Service (Next.js)
```bash
# 前端配置通过 NEXT_PUBLIC_ 前缀自动读取
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_WS_URL=ws://localhost:8080
NEXT_PUBLIC_VOICE_WS_URL=ws://localhost:8001
```

## 🐳 Docker Compose 集成

### 统一配置加载
```yaml
# docker-compose.local.yml
services:
  gateway:
    env_file:
      - .env  # 加载统一配置文件
    environment:
      # 容器内部服务发现配置 (覆盖 .env 中的配置)
      - POSTGRES_HOST=postgres
      - REDIS_HOST=redis
      - PORT=${GATEWAY_PORT:-8080}
      - SERVICE_NAME=${GATEWAY_SERVICE_NAME:-voicehelper-gateway}
```

### 配置优先级
1. **Docker Compose environment** (最高优先级)
2. **Docker Compose env_file** (.env 文件)
3. **Dockerfile ENV** (最低优先级)

## 🧪 配置验证

### 使用验证脚本
```bash
# 运行配置验证
python scripts/validate_env_config.py
```

### 验证输出示例
```
🔍 VoiceHelper 环境配置验证
==================================================
✅ 找到环境配置文件: /path/to/.env
📁 使用配置文件: /path/to/.env

🔧 验证必需配置...
✅ 所有必需配置都已正确设置

🚀 服务配置:
  Gateway:
    端口: 8080
    服务名: voicehelper-gateway
  Algorithm:
    端口: 8000
    服务名: voicehelper-algo

🤖 AI模型配置:
  GLM-4: ✅ 可用
    API地址: https://open.bigmodel.cn/api/paas/v4
  豆包 (ARK): ✅ 可用
    API地址: https://ark.cn-beijing.volces.com/api/v3

📊 配置总结:
✅ 配置验证通过，可以启动服务

🚀 启动命令:
  docker-compose -f docker-compose.local.yml up -d
```

## 🔧 故障排除

### 常见问题

#### 1. 环境变量未生效
```bash
# 检查 .env 文件是否存在
ls -la .env

# 检查 docker-compose 是否正确加载
docker-compose -f docker-compose.local.yml config

# 重新构建容器
docker-compose -f docker-compose.local.yml up --build -d
```

#### 2. API 密钥配置错误
```bash
# 验证 API 密钥格式
python scripts/validate_env_config.py

# 检查容器内环境变量
docker exec voicehelper-algo env | grep GLM_API_KEY
```

#### 3. 端口冲突
```bash
# 检查端口占用
lsof -i :8080
lsof -i :8000

# 修改 .env 文件中的端口配置
GATEWAY_PORT=8081
ALGO_PORT=8001
```

#### 4. 服务无法启动
```bash
# 查看服务日志
docker-compose -f docker-compose.local.yml logs gateway
docker-compose -f docker-compose.local.yml logs algo-service

# 检查健康状态
docker-compose -f docker-compose.local.yml ps
```

## 🚀 部署指南

### 开发环境
```bash
# 1. 创建配置文件
cp env.unified.new .env

# 2. 编辑配置
vim .env

# 3. 验证配置
python scripts/validate_env_config.py

# 4. 启动服务
docker-compose -f docker-compose.local.yml up -d
```

### 生产环境
```bash
# 1. 创建生产配置
cp env.unified.new .env.production

# 2. 修改生产配置
# - 修改所有密码和密钥
# - 设置 ENV=production
# - 设置 GIN_MODE=release
# - 设置 NODE_ENV=production

# 3. 使用生产配置启动
docker-compose -f docker-compose.local.yml --env-file .env.production up -d
```

## 📚 最佳实践

### 1. 安全配置
- ✅ 生产环境必须修改所有默认密码
- ✅ 使用强随机密钥 (JWT_SECRET, ADMIN_SECRET_KEY)
- ✅ 定期轮换 API 密钥
- ✅ 不要将 .env 文件提交到版本控制

### 2. 配置管理
- ✅ 使用配置验证脚本检查配置
- ✅ 为不同环境创建不同的配置文件
- ✅ 使用环境变量覆盖机制
- ✅ 记录配置变更历史

### 3. 监控配置
- ✅ 监控配置文件变更
- ✅ 设置配置错误告警
- ✅ 定期检查配置有效性
- ✅ 备份重要配置文件

## 📞 技术支持

如果遇到配置问题，请：

1. **运行验证脚本**: `python scripts/validate_env_config.py`
2. **查看服务日志**: `docker-compose logs <service-name>`
3. **检查配置文档**: [DEVELOPER_QUICK_START_GUIDE.md](DEVELOPER_QUICK_START_GUIDE.md)
4. **提交 Issue**: [GitHub Issues](https://github.com/example/voicehelper/issues)

---

*最后更新: 2025-09-22*
*文档版本: v1.0.0*
