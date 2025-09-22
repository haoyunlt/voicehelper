# VoiceHelper AI - 本地部署服务 URL 文档

本文档列出了 VoiceHelper AI 系统在本地部署时各个模块的访问地址和接口信息。

## 🌐 服务访问地址总览

### 核心应用服务

| 服务名称 | 端口 | 访问地址 | 状态 | 描述 |
|---------|------|----------|------|------|
| **前端应用** | 3000 | http://localhost:3000 | ✅ 正常 | Next.js 主应用界面 |
| **开发者门户** | 3002 | http://localhost:3002 | ✅ 正常 | API 文档和开发工具 |
| **管理后台** | 5001 | http://localhost:5001 | ⚠️ 待修复 | 系统管理界面 |
| **Nginx 反向代理** | 80/443 | http://localhost | ✅ 正常 | 统一入口，代理所有服务 |

### API 服务

| 服务名称 | 端口 | 访问地址 | 状态 | 描述 |
|---------|------|----------|------|------|
| **算法服务** | 8000 | http://localhost:8000 | ✅ 正常 | AI 核心算法和 RAG 服务 |
| **网关服务** | 8080 | http://localhost:8080 | ✅ 正常 | API 网关和路由 |
| **语音服务** | 8001 | http://localhost:8001 | ✅ 正常 | 语音处理服务 |

### 数据存储服务

| 服务名称 | 端口 | 访问地址 | 状态 | 描述 |
|---------|------|----------|------|------|
| **PostgreSQL** | 5432 | localhost:5432 | ✅ 正常 | 主数据库 |
| **Redis** | 6379 | localhost:6379 | ✅ 正常 | 缓存和会话存储 |
| **Neo4j** | 7474/7687 | http://localhost:7474 | ✅ 正常 | 图数据库 |

### 监控和管理工具

| 服务名称 | 端口 | 访问地址 | 状态 | 描述 |
|---------|------|----------|------|------|
| **pgAdmin** | 5050 | http://localhost:5050 | ✅ 正常 | PostgreSQL 管理界面 |
| **Redis Commander** | 8081 | http://localhost:8081 | ✅ 正常 | Redis 管理界面 |
| **Grafana** | 3004 | http://localhost:3004 | ✅ 正常 | 监控仪表盘 |
| **Prometheus** | 9090 | http://localhost:9090 | ⚠️ 重启中 | 指标收集 |
| **MailHog** | 8025 | http://localhost:8025 | ✅ 正常 | 邮件测试工具 |

---

## 🔧 API 接口文档

### 1. 算法服务 (http://localhost:8000)

#### 核心接口
- **健康检查**: `GET /health`
- **API 文档**: `GET /docs` 
- **OpenAPI 规范**: `GET /openapi.json`

#### 主要功能接口
- **AI 对话查询**: `POST /query`
  ```json
  {
    "messages": [
      {"role": "user", "content": "你的问题"}
    ],
    "temperature": 0.3,
    "max_tokens": 1024
  }
  ```

- **语音查询**: `POST /voice/query`
- **文档摄取**: `POST /ingest`
- **任务状态**: `GET /tasks/{task_id}`
- **取消任务**: `POST /cancel`

#### 测试示例
```bash
# 健康检查
curl http://localhost:8000/health

# AI 对话测试
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好"}]}'
```

### 2. 网关服务 (http://localhost:8080)

#### 核心接口
- **健康检查**: `GET /health`
- **版本信息**: `GET /version`
- **Ping 测试**: `GET /api/v1/ping`
- **错误测试**: `GET /api/v1/error-test`

#### 测试示例
```bash
# 健康检查
curl http://localhost:8080/health

# Ping 测试
curl http://localhost:8080/api/v1/ping
```

### 3. 语音服务 (http://localhost:8001)

#### 核心接口
- **健康检查**: `GET /health`
- **语音查询**: `POST /voice/query`

#### 测试示例
```bash
# 健康检查
curl http://localhost:8001/health
```

---

## 🌍 Nginx 反向代理路由

通过 `http://localhost` 访问的统一入口：

| 路径 | 代理目标 | 描述 |
|------|----------|------|
| `/` | http://frontend:3000 | 前端应用 |
| `/api/` | http://gateway:8080/ | API 网关 |
| `/portal/` | http://developer-portal:3002/ | 开发者门户 |

#### 测试示例
```bash
# 通过 Nginx 访问前端
curl http://localhost/

# 通过 Nginx 访问 API
curl http://localhost/api/health
```

---

## 🔑 数据库连接信息

### PostgreSQL
```
Host: localhost
Port: 5432
Database: voicehelper
Username: voicehelper
Password: voicehelper123
Connection URL: postgresql://voicehelper:voicehelper123@localhost:5432/voicehelper
```

### Redis
```
Host: localhost
Port: 6379
Password: redis123
Connection URL: redis://:redis123@localhost:6379/0
```

### Neo4j
```
HTTP: http://localhost:7474
Bolt: bolt://localhost:7687
Username: neo4j
Password: neo4j123
```

---

## 🛠️ 管理工具访问

### pgAdmin (PostgreSQL 管理)
- **URL**: http://localhost:5050
- **用户名**: admin@voicehelper.ai
- **密码**: admin123

### Redis Commander (Redis 管理)
- **URL**: http://localhost:8081
- **自动连接**: 已配置 Redis 连接

### Grafana (监控仪表盘)
- **URL**: http://localhost:3004
- **用户名**: admin
- **密码**: admin123

### MailHog (邮件测试)
- **URL**: http://localhost:8025
- **SMTP**: localhost:1025

---

## 🚀 快速启动命令

### 启动所有服务
```bash
docker-compose -f docker-compose.local.yml up -d
```

### 查看服务状态
```bash
docker-compose -f docker-compose.local.yml ps
```

### 查看服务日志
```bash
# 查看所有服务日志
docker-compose -f docker-compose.local.yml logs -f

# 查看特定服务日志
docker-compose -f docker-compose.local.yml logs -f algo-service
docker-compose -f docker-compose.local.yml logs -f gateway
docker-compose -f docker-compose.local.yml logs -f frontend
```

### 停止所有服务
```bash
docker-compose -f docker-compose.local.yml down
```

---

## 🔍 健康检查脚本

创建一个简单的健康检查脚本：

```bash
#!/bin/bash
echo "=== VoiceHelper AI 服务健康检查 ==="

services=(
  "算法服务:http://localhost:8000/health"
  "网关服务:http://localhost:8080/health"  
  "语音服务:http://localhost:8001/health"
  "前端服务:http://localhost:3000"
  "Nginx代理:http://localhost"
)

for service in "${services[@]}"; do
  name=$(echo $service | cut -d: -f1)
  url=$(echo $service | cut -d: -f2-)
  
  if curl -f -s "$url" > /dev/null; then
    echo "✅ $name - 正常"
  else
    echo "❌ $name - 异常"
  fi
done
```

---

## 📝 注意事项

1. **端口冲突**: 确保本地没有其他服务占用相同端口
2. **环境变量**: 所有配置通过 `.env` 文件管理，不要硬编码
3. **数据持久化**: 数据库数据存储在 Docker volumes 中
4. **日志查看**: 使用 `docker-compose logs` 命令查看详细日志
5. **服务依赖**: 某些服务有启动顺序依赖，请等待数据库服务完全启动

---

## 🆘 故障排除

### 常见问题

1. **服务无法启动**
   ```bash
   # 检查端口占用
   lsof -i :8000
   
   # 重新构建服务
   docker-compose -f docker-compose.local.yml up -d --build
   ```

2. **数据库连接失败**
   ```bash
   # 检查数据库状态
   docker-compose -f docker-compose.local.yml ps postgres
   
   # 查看数据库日志
   docker-compose -f docker-compose.local.yml logs postgres
   ```

3. **前端页面无法访问**
   ```bash
   # 检查 Nginx 配置
   docker-compose -f docker-compose.local.yml logs nginx
   
   # 直接访问前端服务
   curl http://localhost:3000
   ```

---

*最后更新时间: 2025-09-22*
*文档版本: v1.9.0*
