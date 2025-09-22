# VoiceHelper 使用指南

## 📋 目录

- [项目概述](#项目概述)

- [快速开始](#快速开始)
- [功能介绍](#功能介绍)

- [部署指南](#部署指南)
- [API 使用](#api-使用)

- [前端使用](#前端使用)
- [配置说明](#配置说明)

- [故障排除](#故障排除)
- [最佳实践](#最佳实践)

- [常见问题](#常见问题)

## 🎯 项目概述

VoiceHelper 是一个企业级智能聊天机器人平台，集成了先进的自然语言处理、语音识别、知识检索等技术，为用户提供智能对话服务。

### 核心特性

- 🤖 #### 智能对话: 基于大语言模型的自然对话能力

- 🎤 #### 语音交互: 支持语音输入和语音合成，延迟<150ms
- 📚 #### 知识检索: GraphRAG技术实现精准知识问答

- 🖼️ #### 多模态理解: 图像理解、视频分析、动作识别
- 🌐 #### 全平台支持: Web、移动端、桌面端、微信小程序、浏览器扩展

- 📊 #### 数据分析: 对话数据统计和用户行为分析
- 🔧 #### 易于集成: 提供完整的SDK和API接口

- 🔗 #### 服务集成: 1000+第三方服务集成

### 系统架构

```text
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   前端界面      │    │   后端API       │    │   算法服务      │
│   Next.js       │◄──►│   Go            │◄──►│   Python        │
│   React         │    │   Gin           │    │   FastAPI       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       ▼                       ▼
         │              ┌─────────────────┐    ┌─────────────────┐
         │              │   数据库        │    │   向量数据库    │
         └──────────────┤   Redis         │    │   知识库        │
                        └─────────────────┘    └─────────────────┘
```text

## 🚀 快速开始

### 环境要求

- #### Docker: >= 20.10

- #### Docker Compose: >= 2.0
- #### Node.js: >= 18.0 (开发环境)

- #### Go: >= 1.19 (开发环境)
- #### Python: >= 3.9 (开发环境)

### 一键启动

```bash
# 1. 克隆项目

git clone https://github.com/your-org/voicehelper.git
cd voicehelper

# 2. 启动所有服务

cd deploy
docker-compose up -d

# 3. 等待服务启动完成（约2-3分钟）

docker-compose ps

# 4. 访问应用

# 前端界面: http://localhost:3000
# 后端API: http://localhost:8080

# 算法服务: http://localhost:8000
```text

### 验证安装

```bash
# 检查服务状态

curl http://localhost:8080/health
curl http://localhost:8000/health

# 测试聊天功能

curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好", "user_id": "test_user"}'
```text

## 🎛️ 功能介绍

### 1. 智能对话

#### 核心功能:
- 多轮对话支持

- 上下文理解
- 意图识别

- 情感分析

#### 使用示例:
```javascript
// 发送消息
const response = await fetch('/api/v1/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: '我想了解产品功能',
    user_id: 'user123',
    conversation_id: 'conv456'
  })
});

const result = await response.json();
console.log(result.reply); // AI回复
```text

### 2. 语音交互

#### 支持功能:
- 实时语音识别 (ASR)

- 语音合成 (TTS)
- 多语言支持

- 噪音抑制

#### 前端集成:
```javascript
// 启动语音识别
const startVoiceRecognition = () => {
  const recognition = new webkitSpeechRecognition();
  recognition.lang = 'zh-CN';
  recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    sendMessage(transcript);
  };
  recognition.start();
};
```text

### 3. 知识检索 (RAG)

#### 功能特点:
- 向量相似度检索

- 多文档融合
- 实时知识更新

- 准确性评估

#### 知识库管理:
```bash
# 上传文档

curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@document.pdf" \
  -F "category=product_manual"

# 查询知识

curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "产品如何使用？", "top_k": 5}'
```text

### 4. 数据分析

#### 分析维度:
- 对话量统计

- 用户行为分析
- 满意度评估

- 性能监控

#### 访问方式:
- Web界面: http://localhost:3000/analytics

- API接口: `/api/v1/analytics/*`

## 🐳 部署指南

### Docker Compose 部署（推荐）

#### 1. 准备配置文件
```bash
# 复制环境配置

cp deploy/config/env.local.example deploy/config/env.local

# 编辑配置（根据实际环境修改）

vim deploy/config/env.local
```text

#### 2. 启动服务
```bash
cd deploy
docker-compose up -d

# 查看服务状态

docker-compose ps
docker-compose logs -f
```text

#### 3. 服务端口说明
| 服务 | 端口 | 说明 |
|------|------|------|
| 前端 | 3000 | Web界面 |
| 后端 | 8080 | API服务 |
| 算法 | 8000 | AI服务 |
| PostgreSQL | 5432 | 主数据库 |
| Redis | 6379 | 缓存 |
| MinIO | 9000/9001 | 对象存储 |

### Kubernetes 部署

#### 1. 准备 K8s 配置
```bash
# 应用配置

kubectl apply -f deploy/k8s/

# 检查部署状态

kubectl get pods -n voicehelper
kubectl get services -n voicehelper
```text

#### 2. 配置 Ingress
```yaml
# deploy/k8s/ingress.yaml

apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: voicehelper-ingress
spec:
  rules:
  - host: voicehelper.example.com

    http:
      paths:
      - path: /

        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 3000
      - path: /api

        pathType: Prefix
        backend:
          service:
            name: backend-service
            port:
              number: 8080
```text

### 生产环境部署

#### 1. 环境准备
```bash
# 设置生产环境变量

export NODE_ENV=production
export GIN_MODE=release
export DATABASE_URL=postgresql://user:pass@host:5432/voicehelper
export REDIS_URL=redis://host:6379
```text

#### 2. 安全配置
```bash
# 生成SSL证书

certbot certonly --webroot -w /var/www/html -d voicehelper.example.com

# 配置防火墙

ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 22/tcp
ufw --force enable
```text

#### 3. 监控配置
```bash
# 启动监控服务

docker-compose -f deploy/docker-compose.monitoring.yml up -d

# 访问监控面板

# Grafana: http://localhost:3001
# Prometheus: http://localhost:9090

```text

## 🔌 API 使用

### 认证方式

```bash
# API Key 认证

curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     http://localhost:8080/api/v1/chat
```text

### 核心 API 接口

#### 1. 对话管理

#### 创建对话
```bash
POST /api/v1/conversations
{
  "user_id": "user123",
  "channel": "web",
  "context": {
    "source": "website",
    "page": "product"
  }
}
```text

#### 发送消息
```bash
POST /api/v1/conversations/{conversation_id}/messages
{
  "message": "你好，我想了解产品功能",
  "message_type": "text",
  "metadata": {
    "timestamp": "2025-01-21T10:00:00Z"
  }
}
```text

#### 获取对话历史
```bash
GET /api/v1/conversations/{conversation_id}/messages?limit=20&offset=0
```text

#### 2. 语音处理

#### 语音识别
```bash
POST /api/v1/voice/asr
Content-Type: multipart/form-data

# 上传音频文件

curl -X POST http://localhost:8000/api/v1/voice/asr \
  -F "audio=@voice.wav" \
  -F "language=zh-CN"
```text

#### 语音合成
```bash
POST /api/v1/voice/tts
{
  "text": "你好，欢迎使用VoiceHelper",
  "voice": "zh-CN-XiaoxiaoNeural",
  "speed": 1.0
}
```text

#### 3. 知识管理

#### 上传文档
```bash
POST /api/v1/documents
Content-Type: multipart/form-data

curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@manual.pdf" \
  -F "title=产品手册" \
  -F "category=documentation"
```text

#### 搜索知识
```bash
POST /api/v1/rag/search
{
  "query": "如何重置密码？",
  "top_k": 5,
  "filters": {
    "category": "faq"
  }
}
```text

#### 4. 数据分析

#### 获取统计数据
```bash
GET /api/v1/analytics/conversations?start_date=2025-01-01&end_date=2025-01-31
GET /api/v1/analytics/users?period=7d
GET /api/v1/analytics/performance?metric=response_time
```text

### SDK 使用

#### JavaScript SDK

```bash
npm install @voicehelper/sdk
```text

```javascript
import VoiceHelper from '@voicehelper/sdk';

const client = new VoiceHelper({
  apiKey: 'your-api-key',
  baseURL: 'http://localhost:8080'
});

// 发送消息
const response = await client.chat.send({
  message: '你好',
  userId: 'user123'
});

// 语音识别
const transcript = await client.voice.recognize(audioBlob);

// 知识搜索
const results = await client.knowledge.search('产品功能');
```text

#### Python SDK

```bash
pip install voicehelper-sdk
```text

```python
from voicehelper import VoiceHelperClient

client = VoiceHelperClient(
    api_key='your-api-key',
    base_url='http://localhost:8080'
)

# 发送消息

response = client.chat.send(
    message='你好',
    user_id='user123'
)

# 上传文档

client.documents.upload(
    file_path='document.pdf',
    title='产品手册',
    category='documentation'
)
```text

## 🖥️ 前端使用

### Web 界面功能

#### 1. 聊天界面 (`/chat`)
- 实时对话

- 语音输入
- 文件上传

- 历史记录

#### 2. 知识库管理 (`/datasets`)
- 文档上传

- 分类管理
- 搜索测试

- 质量评估

#### 3. 数据分析 (`/analytics`)
- 对话统计

- 用户分析
- 性能监控

- 趋势图表

### 自定义集成

#### 嵌入式聊天组件
```html
<!-- 在你的网站中嵌入聊天组件 -->
<div id="voicehelper-chat"></div>
<script src="http://localhost:3000/embed.js"></script>
<script>
  VoiceHelper.init({
    container: '#voicehelper-chat',
    apiKey: 'your-api-key',
    theme: 'light',
    language: 'zh-CN'
  });
</script>
```text

#### React 组件
```jsx
import { VoiceChatWidget } from '@voicehelper/react';

function App() {
  return (
    <div>
      <VoiceChatWidget
        apiKey="your-api-key"
        userId="user123"
        onMessage={(message, reply) => {
          console.log('用户:', message);
          console.log('AI:', reply);
        }}
      />
    </div>
  );
}
```text

## ⚙️ 配置说明

### 环境变量配置

#### 后端配置 (`.env`)
```bash
# 服务配置

PORT=8080
GIN_MODE=debug
LOG_LEVEL=info

# 数据库配置

DATABASE_URL=postgresql://user:password@localhost:5432/voicehelper
REDIS_URL=redis://localhost:6379

# AI 服务配置

OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-3.5-turbo

# 向量数据库配置

MILVUS_HOST=localhost
MILVUS_PORT=19530
COLLECTION_NAME=knowledge_base
```text

#### 算法服务配置 (`algo/.env`)
```bash
# 服务配置

HOST=0.0.0.0
PORT=8000
WORKERS=4

# 模型配置

EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIMENSION=1536
MAX_TOKENS=4096

# RAG 配置

CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
SIMILARITY_THRESHOLD=0.7
```text

#### 前端配置 (`frontend/.env.local`)
```bash
# API 配置

NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_WS_URL=ws://localhost:8080

# 功能开关

NEXT_PUBLIC_ENABLE_VOICE=true
NEXT_PUBLIC_ENABLE_FILE_UPLOAD=true
NEXT_PUBLIC_MAX_FILE_SIZE=10485760

# 第三方服务

NEXT_PUBLIC_ANALYTICS_ID=GA_MEASUREMENT_ID
```text

### Docker 配置

#### docker-compose.yml 关键配置
```yaml
services:
  backend:
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/voicehelper

      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres

      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  algo-service:
    environment:

      - MILVUS_PORT=19530
    depends_on:

    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G
```text

## 🔧 故障排除

### 常见问题及解决方案

#### 1. 服务启动失败

#### 问题: Docker 容器无法启动
```bash
# 检查日志

docker-compose logs service-name

# 常见原因及解决方案

# 端口占用
sudo lsof -i :8080
sudo kill -9 PID

# 权限问题

sudo chown -R $USER:$USER ./data
chmod -R 755 ./data

# 内存不足

docker system prune -a
```text

#### 2. 数据库连接问题

#### 问题: 无法连接到 PostgreSQL
```bash
# 检查数据库状态

docker-compose exec postgres pg_isready

# 重置数据库

docker-compose down postgres
docker volume rm deploy_postgres_data
docker-compose up -d postgres

# 手动连接测试

docker-compose exec postgres psql -U postgres -d voicehelper
```text

#### 3. 向量数据库问题

```bash

curl http://localhost:19530/health



# 检查存储空间

df -h
```text

#### 4. AI 服务问题

#### 问题: 算法服务响应慢或失败
```bash
# 检查服务状态

curl http://localhost:8000/health

# 查看资源使用

docker stats algo-service

# 调整配置

# 在 docker-compose.yml 中增加资源限制
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2'
```text

#### 5. 前端问题

#### 问题: 页面加载失败或功能异常
```bash
# 检查前端日志

docker-compose logs frontend

# 重新构建前端

docker-compose build --no-cache frontend
docker-compose up -d frontend

# 清除浏览器缓存

# Chrome: Ctrl+Shift+R
# 或在开发者工具中禁用缓存

```text

### 性能优化

#### 1. 数据库优化

```sql
-- 创建索引
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at);

-- 分析查询性能
EXPLAIN ANALYZE SELECT * FROM messages WHERE conversation_id = 'xxx';
```text

#### 2. 缓存优化

```bash
# Redis 配置优化

# 在 redis.conf 中设置
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
```text

#### 3. 应用优化

#### 后端优化
```go
// 连接池配置
db.SetMaxOpenConns(25)
db.SetMaxIdleConns(25)
db.SetConnMaxLifetime(5 * time.Minute)

// 启用 gzip 压缩
router.Use(gin.Recovery())
router.Use(gzip.Gzip(gzip.DefaultCompression))
```text

#### 前端优化
```javascript
// next.config.js
module.exports = {
  compress: true,
  images: {
    domains: ['localhost'],
    formats: ['image/webp', 'image/avif'],
  },
  experimental: {
    optimizeCss: true,
  }
};
```text

## 📚 最佳实践

### 1. 安全最佳实践

#### API 安全
```bash
# 使用 HTTPS

# 配置 SSL 证书
# 实施 API 限流

# 启用 CORS 保护
# 使用强密码和 API Key

```text

#### 数据安全
```bash
# 数据库加密

# 敏感信息脱敏
# 定期备份

# 访问控制
# 审计日志

```text

### 2. 性能最佳实践

#### 系统监控
```bash
# 设置监控指标

# 配置告警规则
# 定期性能测试

# 容量规划
# 故障预案

```text

#### 资源优化
```bash
# 合理设置资源限制

# 使用缓存策略
# 数据库查询优化

# 静态资源 CDN
# 负载均衡

```text

### 3. 运维最佳实践

#### 部署策略
```bash
# 蓝绿部署

# 滚动更新
# 健康检查

# 回滚机制
# 配置管理

```text

#### 备份恢复
```bash
# 定期备份脚本

#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec postgres pg_dump -U postgres voicehelper > backup_${DATE}.sql
aws s3 cp backup_${DATE}.sql s3://voicehelper-backups/
```text

## ❓ 常见问题

### Q1: 如何添加新的语言支持？

#### A1:
1. 在算法服务中添加对应的语言模型

2. 更新前端语言配置
3. 添加语言包翻译文件

4. 测试语音识别和合成功能

### Q2: 如何扩展知识库？

#### A2:
1. 通过 Web 界面上传文档

2. 使用 API 批量导入
3. 配置自动同步脚本

4. 定期更新和维护

### Q3: 如何集成到现有系统？

#### A3:
1. 使用 REST API 接口

2. 集成 JavaScript SDK
3. 嵌入聊天组件

4. 配置 SSO 认证

### Q4: 如何监控系统性能？

#### A4:
1. 使用内置监控面板

2. 配置 Prometheus + Grafana
3. 设置告警规则

4. 定期性能测试

### Q5: 如何处理高并发？

#### A5:
1. 水平扩展服务实例

2. 使用负载均衡
3. 优化数据库查询

4. 增加缓存层

---

## 📞 技术支持

- #### 文档: [https://docs.voicehelper.com](https://docs.voicehelper.com)

- #### GitHub: [https://github.com/your-org/voicehelper](https://github.com/your-org/voicehelper)
- #### 问题反馈: [GitHub Issues](https://github.com/your-org/voicehelper/issues)

- #### 技术交流: [Discord 社区](https://discord.gg/voicehelper)

## 📄 许可证

本项目采用 MIT 许可证，详见 [LICENSE](../LICENSE) 文件。

---

#### 最后更新: 2025-09-22
#### 文档版本: v1.0
#### 适用系统版本: v1.20.0+
