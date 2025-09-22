# VoiceHelper 架构设计文档

## 系统概览

VoiceHelper 是一个多模态 AI 助手平台，支持语音、文本、图像交互，基于微服务架构构建。

### 核心能力
- **多模态交互**：语音识别/合成、文本对话、图像理解
- **实时通信**：WebSocket、SSE、WebRTC 支持
- **AI 编排**：RAG 检索增强、Agent 工作流、多模型路由
- **多平台支持**：Web、移动端、桌面应用、浏览器扩展

## 技术架构

### 系统分层
```
┌─────────────────────────────────────────────────────────┐
│                    客户端层                              │
│  Web App │ Mobile │ Desktop │ Browser Extension │ API   │
├─────────────────────────────────────────────────────────┤
│                    网关层                                │
│           Go Gateway (gRPC + HTTP)                      │
├─────────────────────────────────────────────────────────┤
│                   服务层                                 │
│  Chat Service │ Voice Service │ RAG Service │ AI Service │
├─────────────────────────────────────────────────────────┤
│                   数据层                                 │
│   PostgreSQL │ Redis │ Vector DB │ Object Storage      │
└─────────────────────────────────────────────────────────┘
```

### 核心服务

#### 1. Gateway 网关 (Go)
- **职责**：路由、认证、限流、协议转换
- **端口**：8080 (HTTP), 8081 (gRPC)
- **特性**：
  - 统一 API 入口
  - WebSocket/SSE 连接管理
  - 请求/响应转换
  - 监控埋点

#### 2. AI 算法服务 (Python)
- **职责**：LLM 调用、RAG 检索、推理编排
- **端口**：8000 (FastAPI)
- **核心模块**：
  - `core/llm/`: 多模型适配 (OpenAI/Anthropic/国产)
  - `core/rag/`: BGE + FAISS 向量检索
  - `core/agents/`: 工作流编排
  - `reasoning/`: 逻辑/数学/因果推理

#### 3. 语音服务 (Python)
- **职责**：ASR/TTS、实时语音处理
- **端口**：8002 (FastAPI)
- **技术栈**：
  - ASR: Whisper/Azure Speech
  - TTS: Azure/ElevenLabs
  - 实时处理: WebRTC

#### 4. 数据存储
- **PostgreSQL**: 用户数据、会话记录
- **Redis**: 缓存、会话状态
- **Vector DB**: 知识库向量索引
- **S3**: 文件存储

## 关键技术决策

### 1. 多模型路由策略
```python
# 按成本/质量/延迟路由
routing_config = {
    "chat": {
        "primary": "gpt-4o-mini",      # 性价比
        "fallback": "claude-3-haiku",   # 备用
        "premium": "gpt-4o"            # 高质量
    },
    "reasoning": {
        "primary": "claude-3.5-sonnet", # 推理能力
        "fallback": "gpt-4o"
    }
}
```

### 2. RAG 检索流程
```
文档 → 清洗 → 分块 → BGE嵌入 → FAISS索引
查询 → BGE嵌入 → 相似度检索 → 重排序 → 上下文注入
```

### 3. 实时通信架构
- **WebSocket**: 双向实时消息
- **SSE**: 服务端推送 (LLM 流式输出)
- **WebRTC**: P2P 语音通话

## 性能指标 (SLO)

| 指标 | 目标 | 监控 |
|------|------|------|
| API 响应时间 P95 | < 200ms | Prometheus |
| 端到端对话 P95 | < 2.5s | 链路追踪 |
| 系统可用性 | ≥ 99.9% | 健康检查 |
| 错误率 | < 0.1% | 日志聚合 |

## 安全与合规

### 数据保护
- **传输加密**: TLS 1.3
- **存储加密**: AES-256
- **密钥管理**: KMS 轮换
- **PII 脱敏**: 自动检测和掩码

### 访问控制
- **认证**: JWT + OAuth2
- **授权**: RBAC 角色权限
- **审计**: 操作日志记录

## 部署架构

### 容器化部署
```yaml
# docker-compose 核心服务
services:
  gateway:
    image: voicehelper/gateway:latest
    ports: ["8080:8080", "8081:8081"]
  
  ai-service:
    image: voicehelper/ai:latest
    ports: ["8000:8000"]
    
  voice-service:
    image: voicehelper/voice:latest
    ports: ["8002:8002"]
```

### 扩展策略
- **水平扩展**: K8s HPA 基于 CPU/内存
- **垂直扩展**: GPU 节点池 (推理加速)
- **缓存策略**: Redis 集群 + 语义缓存

## 监控观测

### 指标收集
- **系统指标**: CPU、内存、网络、磁盘
- **业务指标**: QPS、延迟、错误率、成本
- **AI 指标**: Token 消耗、模型延迟、准确率

### 告警策略
```yaml
alerts:
  - name: high_latency
    condition: p95_latency > 2s
    action: 自动扩容 + 通知
    
  - name: error_spike
    condition: error_rate > 1%
    action: 熔断 + 降级
```

## 成本优化

### Token 经济
- **缓存策略**: 语义缓存 + 提示缓存
- **模型选择**: 按任务复杂度路由
- **上下文压缩**: 动态截断 + 关键信息保留

### 资源优化
- **按需扩容**: 基于负载自动调整
- **GPU 共享**: 多租户推理服务
- **存储分层**: 热数据 SSD + 冷数据对象存储

---

*最后更新: 2025-09-22*
*版本: v2.0*
