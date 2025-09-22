# 环境变量和配置指南

## 📋 概述

本文档详细说明了 VoiceHelper 项目的所有环境变量和配置项，包括必需配置、可选配置以及不同环境下的配置建议。

## 🔑 核心必要环境变量

### 豆包 API 配置（必填）

```bash
# ⚠️ 必须配置，否则AI功能无法使用
ARK_API_KEY=your_ark_api_key_here
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL=ep-20241201140014-vbzjz
```

**说明**:
- `ARK_API_KEY`: 豆包API密钥，**必须填写**
- `ARK_BASE_URL`: 豆包API基础URL，一般无需修改
- `ARK_MODEL`: 使用的模型ID，根据实际申请的模型修改

### 数据库连接（必需）

#### PostgreSQL 主数据库
```bash
# 推荐使用 DATABASE_URL 统一配置
DATABASE_URL=postgres://chatbot:chatbot123@localhost:5432/chatbot?sslmode=disable

# 或者分别配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=chatbot
POSTGRES_USER=chatbot
POSTGRES_PASSWORD=chatbot123  # ⚠️ 生产环境必须修改
```

#### Redis 缓存数据库
```bash
# 推荐使用 REDIS_URL 统一配置
REDIS_URL=redis://localhost:6379

# 或者分别配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=redis123       # ⚠️ 生产环境必须修改
REDIS_DB=0
```

### 向量数据库 Milvus（必需）

```bash
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_USER=                  # 可选，无用户名时留空
MILVUS_PASSWORD=              # 可选，无密码时留空
```

## 🎯 服务配置

### 端口配置

```bash
BACKEND_PORT=8080             # Go后端服务端口
FRONTEND_PORT=3000            # Next.js前端端口
ALGO_PORT=8000                # Python算法服务端口
PORT=8080                     # 通用端口变量（后端使用）
```

### 运行环境

```bash
NODE_ENV=development          # 前端环境: production | development
GIN_MODE=debug                # Go服务模式: release | debug
ENV=development               # 通用环境变量: production | development
LOG_LEVEL=debug               # 日志级别: debug | info | warn | error
```

## 🧠 AI/ML 配置

### Embedding 模型配置

```bash
EMBEDDING_MODEL=bge-m3        # 嵌入模型名称
EMBEDDING_DIMENSION=1024      # 向量维度，需与模型匹配
```

## 🔐 安全配置

### JWT 和密钥管理

```bash
# ⚠️ 生产环境必须使用强随机字符串
JWT_SECRET=your-jwt-secret-key-for-development
ADMIN_SECRET_KEY=your-admin-secret-key-for-development
```

**安全要求**:
- 密钥长度至少32字符
- 使用随机生成器生成
- 不同环境使用不同密钥
- 定期更换

### 微信小程序配置（可选）

```bash
WECHAT_APP_ID=your-wechat-app-id
WECHAT_APP_SECRET=your-wechat-app-secret
```

## 🗄️ 扩展数据库（可选）

### Neo4j 图数据库

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=neo4j123
```

**用途**: 知识图谱、关系推理等高级RAG功能

## 📊 监控和观测性（推荐配置）

### Prometheus 和 Grafana

```bash
PROMETHEUS_ADDR=localhost:9090
GRAFANA_ADDR=localhost:3001
```

### 日志收集

```bash
ELASTICSEARCH_HOST=localhost
ELASTICSEARCH_PORT=9200
```

### 消息队列

```bash
RABBITMQ_HOST=localhost
RABBITMQ_PORT=5672
RABBITMQ_USER=rabbitmq
RABBITMQ_PASSWORD=rabbitmq123
```

### 文件存储

```bash
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
```

## 🌐 服务发现和通信

### 内部服务地址

```bash
ALGO_SERVICE_URL=http://localhost:8000     # 后端调用算法服务
ADMIN_SERVICE_URL=http://localhost:5001    # 管理后台地址
```

### 前端配置

```bash
NEXT_PUBLIC_API_URL=http://localhost:8080  # 前端调用后端API
NEXT_PUBLIC_WS_URL=ws://localhost:8080     # WebSocket连接地址
```

## 📁 配置文件说明

### 配置文件位置

| 文件路径 | 说明 | 用途 |
|----------|------|------|
| `/env.example` | 主配置模板 | 根目录环境变量模板 |
| `/algo/env.example` | 算法服务配置模板 | Python服务专用配置 |
| `/deploy/config/env.local` | 完整开发环境配置 | 本地开发完整配置参考 |
| `/algo/core/config.py` | Python配置类 | 代码中的配置管理 |

### Docker Compose 配置

- `docker-compose.yml`: 生产环境配置
- `docker-compose.local.yml`: 本地开发完整配置
- `deploy/compose/docker-compose.dev.yml`: 开发环境配置

## 🚀 快速启动指南

### 1. 环境变量设置

```bash
# 复制配置模板
cp env.example .env

# 编辑配置文件
vim .env
```

### 2. 必须配置的项目

**最小启动配置**:
```bash
# 必填
ARK_API_KEY=your_actual_api_key_here

# 数据库（使用默认配置或修改）
DATABASE_URL=postgres://chatbot:chatbot123@localhost:5432/chatbot?sslmode=disable
REDIS_URL=redis://localhost:6379

# Milvus（使用默认配置）
MILVUS_HOST=localhost
MILVUS_PORT=19530
```

### 3. 启动服务

```bash
# 使用 Makefile
make up

# 或直接使用 docker-compose
docker-compose up -d
```

## 🔧 不同环境配置建议

### 开发环境 (development)

```bash
ENV=development
NODE_ENV=development
GIN_MODE=debug
LOG_LEVEL=debug

# 可以使用默认密码
POSTGRES_PASSWORD=chatbot123
REDIS_PASSWORD=redis123
JWT_SECRET=development-secret-key
```

### 测试环境 (testing)

```bash
ENV=testing
NODE_ENV=production
GIN_MODE=release
LOG_LEVEL=info

# 使用测试专用密码
POSTGRES_PASSWORD=test_secure_password
REDIS_PASSWORD=test_redis_password
JWT_SECRET=testing-jwt-secret-key
```

### 生产环境 (production)

```bash
ENV=production
NODE_ENV=production
GIN_MODE=release
LOG_LEVEL=warn

# ⚠️ 必须使用强密码和随机密钥
POSTGRES_PASSWORD=your_very_secure_postgres_password
REDIS_PASSWORD=your_very_secure_redis_password
JWT_SECRET=your_very_long_random_jwt_secret_key
ADMIN_SECRET_KEY=your_very_long_random_admin_secret_key
```

## ⚠️ 安全注意事项

### 密码安全

1. **永远不要**将真实的API密钥和密码提交到代码仓库
2. 生产环境**必须**修改所有默认密码
3. 使用环境变量或密钥管理服务存储敏感信息
4. 定期轮换密码和密钥

### 推荐的密钥生成方法

```bash
# 生成随机JWT密钥
openssl rand -hex 32

# 生成随机密码
openssl rand -base64 32
```

### 权限最小化

- 数据库用户只分配必要的权限
- Redis 启用认证并限制命令
- 使用防火墙限制服务访问

## 🐛 故障排查

### 常见问题

1. **API调用失败**
   - 检查 `ARK_API_KEY` 是否正确设置
   - 验证网络连接和API额度

2. **数据库连接失败**
   - 检查数据库服务是否启动
   - 验证连接字符串格式和凭据

3. **Milvus连接问题**
   - 确认Milvus服务状态
   - 检查端口是否被占用

4. **前端无法访问API**
   - 验证 `NEXT_PUBLIC_API_URL` 配置
   - 检查CORS设置

### 日志查看

```bash
# 查看所有服务日志
docker-compose logs

# 查看特定服务日志
docker-compose logs backend
docker-compose logs algo-service
```

## 📚 相关文档

- [开发指南](./DEVELOPMENT_GUIDE.md)
- [端口使用说明](./PORT_USAGE.md)
- [部署指南](../deploy/README.md)
- [API文档](./api/openapi_v3.yaml)

---

**最后更新**: 2024-12-21  
**维护者**: VoiceHelper Team
