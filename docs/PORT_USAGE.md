# 🔌 端口占用文档 - VoiceHelper AI

## 📋 概述

本文档详细说明了VoiceHelper AI智能对话系统中各个服务和组件使用的端口配置，包括默认端口、可配置端口以及端口冲突解决方案。

---

## 🌐 核心服务端口

### 应用服务层

| 服务名称 | 默认端口 | 协议 | 用途 | 配置文件 | 环境变量 |
|---------|---------|------|------|----------|----------|
| **Web前端** | 3000 | HTTP/HTTPS | Next.js前端应用 | `frontend/package.json` | `FRONTEND_PORT` |
| **开发者门户** | 3002 | HTTP/HTTPS | 开发者平台和SDK | `developer-portal/package.json` | `PORTAL_PORT` |
| **API网关** | 8080 | HTTP/HTTPS | Go后端服务网关 | `backend/cmd/server/main.go` | `GATEWAY_PORT` |
| **算法服务** | 8000 | HTTP/HTTPS | Python FastAPI服务 | `algo/app/main.py` | `ALGO_PORT` |
| **语音服务** | 8001 | HTTP/WebSocket | 语音处理服务 | `algo/app/voice_server.py` | `VOICE_PORT` |
| **管理后台** | 5001 | HTTP/HTTPS | 运营管理界面 | `admin/app.py` | `ADMIN_PORT` |

### WebSocket服务

| 服务名称 | 默认端口 | 协议 | 用途 | 路径 |
|---------|---------|------|------|------|
| **实时对话** | 8080 | WebSocket | 实时文本对话 | `/api/v1/chat/stream` |
| **语音流** | 8001 | WebSocket | 实时语音处理 | `/api/v1/voice/stream` |
| **系统通知** | 8080 | WebSocket | 系统状态通知 | `/api/v1/notifications` |

---

## 🗄️ 数据存储服务端口

### 数据库服务

| 服务名称 | 默认端口 | 协议 | 用途 | Docker服务名 | 环境变量 |
|---------|---------|------|------|-------------|----------|
| **PostgreSQL** | 5432 | TCP | 主数据库 | `postgres` | `POSTGRES_PORT` |
| **Redis** | 6379 | TCP | 缓存和会话 | `redis` | `REDIS_PORT` |
| **Milvus** | 19530 | gRPC | 向量数据库 | `milvus-standalone` | `MILVUS_PORT` |
| **Neo4j** | 7474 | HTTP | 图数据库Web界面 | `neo4j` | `NEO4J_HTTP_PORT` |
| **Neo4j Bolt** | 7687 | Bolt | 图数据库连接 | `neo4j` | `NEO4J_BOLT_PORT` |

### 数据库管理界面

| 服务名称 | 默认端口 | 协议 | 用途 | 访问地址 |
|---------|---------|------|------|----------|
| **pgAdmin** | 5050 | HTTP | PostgreSQL管理 | `http://localhost:5050` |
| **Redis Commander** | 8081 | HTTP | Redis管理 | `http://localhost:8081` |
| **Milvus Attu** | 3001 | HTTP | Milvus管理 | `http://localhost:3001` |
| **Neo4j Browser** | 7474 | HTTP | Neo4j管理 | `http://localhost:7474` |

---

## 📊 监控和运维服务端口

### 监控服务

| 服务名称 | 默认端口 | 协议 | 用途 | 配置文件 | 环境变量 |
|---------|---------|------|------|----------|----------|
| **Prometheus** | 9090 | HTTP | 指标收集 | `deploy/config/prometheus.yml` | `PROMETHEUS_PORT` |
| **Grafana** | 3001 | HTTP | 数据可视化 | `deploy/config/grafana.ini` | `GRAFANA_PORT` |
| **Jaeger** | 16686 | HTTP | 链路追踪UI | `deploy/config/jaeger.yml` | `JAEGER_PORT` |
| **Jaeger Collector** | 14268 | HTTP | 链路数据收集 | `deploy/config/jaeger.yml` | `JAEGER_COLLECTOR_PORT` |
| **AlertManager** | 9093 | HTTP | 告警管理 | `deploy/config/alertmanager.yml` | `ALERTMANAGER_PORT` |

### 日志服务

| 服务名称 | 默认端口 | 协议 | 用途 | 配置文件 |
|---------|---------|------|------|----------|
| **Elasticsearch** | 9200 | HTTP | 日志存储 | `deploy/config/elasticsearch.yml` |
| **Kibana** | 5601 | HTTP | 日志分析 | `deploy/config/kibana.yml` |
| **Logstash** | 5044 | TCP | 日志收集 | `deploy/config/logstash.conf` |
| **Fluentd** | 24224 | TCP | 日志转发 | `deploy/config/fluentd.conf` |

---

## 🔧 开发和测试服务端口

### 开发服务

| 服务名称 | 默认端口 | 协议 | 用途 | 启动命令 |
|---------|---------|------|------|----------|
| **前端开发服务器** | 3000 | HTTP | Next.js开发模式 | `npm run dev` |
| **后端开发服务器** | 8080 | HTTP | Go开发模式 | `go run cmd/server/main.go` |
| **算法开发服务器** | 8000 | HTTP | FastAPI开发模式 | `uvicorn app.main:app --reload` |
| **Storybook** | 6006 | HTTP | 组件开发 | `npm run storybook` |

### 测试服务

| 服务名称 | 默认端口 | 协议 | 用途 | 配置 |
|---------|---------|------|------|------|
| **测试数据库** | 5433 | TCP | PostgreSQL测试 | `TEST_POSTGRES_PORT` |
| **测试Redis** | 6380 | TCP | Redis测试 | `TEST_REDIS_PORT` |
| **Mock服务** | 8888 | HTTP | API模拟 | `tests/mock/server.js` |
| **性能测试** | 8889 | HTTP | 负载测试 | `tests/performance/locust.py` |

---

## 🐳 Docker容器端口映射

### 生产环境 (docker-compose.yml)

```yaml
services:
  frontend:
    ports:
      - "3000:3000"
  
  gateway:
    ports:
      - "8080:8080"
  
  algo-service:
    ports:
      - "8000:8000"
  
  voice-service:
    ports:
      - "8001:8001"
  
  admin:
    ports:
      - "5001:5001"
  
  postgres:
    ports:
      - "5432:5432"
  
  redis:
    ports:
      - "6379:6379"
  
  milvus:
    ports:
      - "19530:19530"
  
  neo4j:
    ports:
      - "7474:7474"
      - "7687:7687"
  
  prometheus:
    ports:
      - "9090:9090"
  
  grafana:
    ports:
      - "3001:3000"
```

### 开发环境 (docker-compose.local.yml)

```yaml
services:
  # 开发环境端口映射
  frontend-dev:
    ports:
      - "3000:3000"
      - "3001:3001"  # HMR端口
  
  # 数据库管理工具
  pgadmin:
    ports:
      - "5050:80"
  
  redis-commander:
    ports:
      - "8081:8081"
  
  # 开发工具
  storybook:
    ports:
      - "6006:6006"
```

---

## ⚙️ 端口配置

### 环境变量配置

```bash
# .env 文件示例

# 应用服务端口
FRONTEND_PORT=3000
PORTAL_PORT=3002
GATEWAY_PORT=8080
ALGO_PORT=8000
VOICE_PORT=8001
ADMIN_PORT=5001

# 数据库端口
POSTGRES_PORT=5432
REDIS_PORT=6379
MILVUS_PORT=19530
NEO4J_HTTP_PORT=7474
NEO4J_BOLT_PORT=7687

# 监控服务端口
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
JAEGER_PORT=16686

# 开发工具端口
STORYBOOK_PORT=6006
MOCK_SERVER_PORT=8888

# 测试环境端口
TEST_POSTGRES_PORT=5433
TEST_REDIS_PORT=6380
```

### 配置文件示例

#### Go服务端口配置 (backend/config/config.go)

```go
type Config struct {
    Server struct {
        Port     int    `env:"GATEWAY_PORT" envDefault:"8080"`
        Host     string `env:"GATEWAY_HOST" envDefault:"0.0.0.0"`
        Protocol string `env:"GATEWAY_PROTOCOL" envDefault:"http"`
    }
    
    Database struct {
        PostgresPort int `env:"POSTGRES_PORT" envDefault:"5432"`
        RedisPort    int `env:"REDIS_PORT" envDefault:"6379"`
        MilvusPort   int `env:"MILVUS_PORT" envDefault:"19530"`
        Neo4jPort    int `env:"NEO4J_BOLT_PORT" envDefault:"7687"`
    }
}
```

#### Python服务端口配置 (algo/config/settings.py)

```python
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    # 服务端口
    algo_port: int = int(os.getenv("ALGO_PORT", 8000))
    voice_port: int = int(os.getenv("VOICE_PORT", 8001))
    
    # 数据库端口
    postgres_port: int = int(os.getenv("POSTGRES_PORT", 5432))
    redis_port: int = int(os.getenv("REDIS_PORT", 6379))
    milvus_port: int = int(os.getenv("MILVUS_PORT", 19530))
    
    # 监控端口
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", 9090))
    
    class Config:
        env_file = ".env"
```

#### Next.js端口配置 (frontend/next.config.js)

```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  // 开发服务器配置
  devServer: {
    port: process.env.FRONTEND_PORT || 3000,
    host: '0.0.0.0'
  },
  
  // API代理配置
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `http://localhost:${process.env.GATEWAY_PORT || 8080}/api/:path*`
      }
    ]
  }
}

module.exports = nextConfig
```

---

## 🔍 端口检查和管理

### 端口占用检查

```bash
# 检查特定端口占用
lsof -i :3000
netstat -tulpn | grep :3000

# 检查所有VoiceHelper相关端口
for port in 3000 3001 3002 5001 8000 8001 8080 5432 6379 19530 7474 7687 9090; do
    echo "检查端口 $port:"
    lsof -i :$port
    echo "---"
done
```

### 端口释放脚本

```bash
#!/bin/bash
# scripts/kill-ports.sh

# VoiceHelper AI 端口清理脚本
PORTS=(3000 3001 3002 5001 8000 8001 8080 5432 6379 19530 7474 7687 9090)

echo "🔌 清理VoiceHelper AI端口占用..."

for port in "${PORTS[@]}"; do
    echo "检查端口 $port..."
    PID=$(lsof -ti:$port)
    
    if [ ! -z "$PID" ]; then
        echo "发现端口 $port 被进程 $PID 占用，正在终止..."
        kill -9 $PID
        echo "✅ 端口 $port 已释放"
    else
        echo "✅ 端口 $port 未被占用"
    fi
done

echo "🎉 端口清理完成！"
```

### 端口健康检查脚本

```bash
#!/bin/bash
# scripts/check-ports.sh

# VoiceHelper AI 端口健康检查脚本
declare -A SERVICES=(
    ["3000"]="前端服务"
    ["3002"]="开发者门户"
    ["8080"]="API网关"
    ["8000"]="算法服务"
    ["8001"]="语音服务"
    ["5001"]="管理后台"
    ["5432"]="PostgreSQL"
    ["6379"]="Redis"
    ["19530"]="Milvus"
    ["7474"]="Neo4j HTTP"
    ["7687"]="Neo4j Bolt"
    ["9090"]="Prometheus"
    ["3001"]="Grafana"
)

echo "🔍 VoiceHelper AI 服务端口健康检查"
echo "======================================"

for port in "${!SERVICES[@]}"; do
    service_name="${SERVICES[$port]}"
    
    if nc -z localhost $port 2>/dev/null; then
        echo "✅ $service_name (端口 $port): 正常运行"
    else
        echo "❌ $service_name (端口 $port): 服务未启动"
    fi
done

echo "======================================"
echo "检查完成！"
```

---

## 🚨 端口冲突解决方案

### 常见端口冲突

| 默认端口 | 可能冲突的服务 | 解决方案 |
|---------|---------------|----------|
| **3000** | React开发服务器, Grafana | 使用环境变量 `FRONTEND_PORT=3003` |
| **3001** | Grafana默认端口 | 使用环境变量 `GRAFANA_PORT=3004` |
| **5432** | 其他PostgreSQL实例 | 使用环境变量 `POSTGRES_PORT=5433` |
| **6379** | 其他Redis实例 | 使用环境变量 `REDIS_PORT=6380` |
| **8080** | Tomcat, Jenkins | 使用环境变量 `GATEWAY_PORT=8082` |
| **9090** | 其他Prometheus实例 | 使用环境变量 `PROMETHEUS_PORT=9091` |

### 端口冲突检测脚本

```bash
#!/bin/bash
# scripts/detect-conflicts.sh

echo "🔍 检测端口冲突..."

# 检查关键端口是否被其他服务占用
CRITICAL_PORTS=(3000 8080 8000 5432 6379)

for port in "${CRITICAL_PORTS[@]}"; do
    if lsof -i :$port >/dev/null 2>&1; then
        echo "⚠️  端口 $port 已被占用:"
        lsof -i :$port
        echo "建议使用备用端口或停止占用服务"
        echo "---"
    fi
done
```

### 动态端口分配

```python
# utils/port_manager.py
import socket
from contextlib import closing

def find_free_port(start_port=8000, max_attempts=100):
    """查找可用端口"""
    for port in range(start_port, start_port + max_attempts):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(('localhost', port)) != 0:
                return port
    raise RuntimeError(f"无法在 {start_port}-{start_port + max_attempts} 范围内找到可用端口")

def is_port_available(port):
    """检查端口是否可用"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex(('localhost', port)) != 0
```

---

## 🔒 安全配置

### 防火墙配置

```bash
# Ubuntu/Debian 防火墙配置
sudo ufw allow 3000/tcp comment "VoiceHelper Frontend"
sudo ufw allow 8080/tcp comment "VoiceHelper Gateway"
sudo ufw allow 8000/tcp comment "VoiceHelper Algo"

# 仅允许本地访问的数据库端口
sudo ufw allow from 127.0.0.1 to any port 5432 comment "PostgreSQL Local"
sudo ufw allow from 127.0.0.1 to any port 6379 comment "Redis Local"
```

### Nginx反向代理配置

```nginx
# /etc/nginx/sites-available/voicehelper
server {
    listen 80;
    server_name voicehelper.ai;
    
    # 前端服务
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # API网关
    location /api/ {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    
    # WebSocket支持
    location /api/v1/voice/stream {
        proxy_pass http://localhost:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 📋 端口使用清单

### 生产环境端口清单

```
🌐 用户访问端口:
├── 80/443    - Nginx反向代理 (HTTP/HTTPS)
├── 3000      - Web前端服务
├── 3002      - 开发者门户
└── 5001      - 管理后台

🔧 API服务端口:
├── 8080      - API网关 (主要入口)
├── 8000      - 算法服务 (FastAPI)
└── 8001      - 语音服务 (WebSocket)

🗄️ 数据存储端口:
├── 5432      - PostgreSQL (主数据库)
├── 6379      - Redis (缓存)
├── 19530     - Milvus (向量数据库)
├── 7474      - Neo4j HTTP (图数据库)
└── 7687      - Neo4j Bolt (图数据库连接)

📊 监控服务端口:
├── 9090      - Prometheus (指标收集)
├── 3001      - Grafana (数据可视化)
├── 16686     - Jaeger (链路追踪)
└── 9093      - AlertManager (告警管理)
```

### 开发环境额外端口

```
🛠️ 开发工具端口:
├── 6006      - Storybook (组件开发)
├── 8888      - Mock服务 (API模拟)
├── 5050      - pgAdmin (数据库管理)
├── 8081      - Redis Commander (Redis管理)
└── 5601      - Kibana (日志分析)

🧪 测试环境端口:
├── 5433      - 测试PostgreSQL
├── 6380      - 测试Redis
└── 8889      - 性能测试服务
```

---

## 📞 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 查找占用进程
   lsof -i :端口号
   # 终止进程
   kill -9 PID
   ```

2. **服务无法启动**
   ```bash
   # 检查端口可用性
   nc -z localhost 端口号
   # 检查防火墙设置
   sudo ufw status
   ```

3. **WebSocket连接失败**
   ```bash
   # 检查WebSocket端口
   curl -i -N -H "Connection: Upgrade" -H "Upgrade: websocket" http://localhost:8001/api/v1/voice/stream
   ```

### 联系支持

- **技术文档**: [https://docs.voicehelper.ai](https://docs.voicehelper.ai)
- **GitHub Issues**: [提交端口相关问题](https://github.com/voicehelper/voicehelper/issues)
- **开发者社区**: [Discord](https://discord.gg/voicehelper)

---

*最后更新: 2025-09-22*  
*版本: v1.9.0*  
*维护者: VoiceHelper运维团队*
