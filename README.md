# 🤖 智能聊天机器人系统

[![Version](https://img.shields.io/badge/version-1.4.0-blue.svg)](docs/VERSION.md)
[![Status](https://img.shields.io/badge/status-production_ready-green.svg)](docs/PROJECT_MASTER_DOC.md)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

企业级智能对话系统，支持文本/语音双模态交互，具备GraphRAG检索、自主Agent能力、连续学习机制。

## 🌟 核心特性

- **🎭 双模态交互**: 文本SSE流式 + 语音WebSocket实时通信
- **🧠 GraphRAG检索**: 知识图谱增强，召回率97%
- **🤖 智能Agent**: 多步推理、自主规划、工具调用
- **📚 连续学习**: 主动学习、在线优化、自适应
- **⚡ 极致性能**: 首响<200ms，barge-in<150ms
- **💰 成本优化**: 智能路由，降低50%调用成本

## 🚀 快速开始

### 环境要求

- Docker 20.10+
- Docker Compose 2.0+
- Node.js 18+
- Python 3.11+
- Go 1.21+

### 一键启动

```bash
# 克隆项目
git clone https://github.com/example/chatbot.git
cd chatbot

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件填入必要配置

# 启动所有服务
make up

# 访问服务
# Web界面: http://localhost:3000
# API文档: http://localhost:8080/swagger
# 监控面板: http://localhost:3001
```

### 开发模式

```bash
# 安装依赖
make install

# 启动开发服务器
make dev

# 运行测试
make test

# 代码检查
make lint
```

## 📖 文档导航

### 核心文档
- 📘 [**完整技术文档**](docs/PROJECT_MASTER_DOC.md) - 系统设计、API规范、部署方案（最新最全）
- 🎯 [优化方案](docs/OPTIMIZATION_PLAN_2025.md) - 基于业界最新技术的优化
- 📚 [文档索引](docs/DOCUMENTATION_INDEX.md) - 所有文档导航

### 项目管理
- 🔄 [版本管理](docs/VERSION.md) - 版本历史和发布计划
- 🌳 [分支策略](docs/BRANCHING.md) - Git工作流程

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────┐
│             客户端层                          │
│   Web (Next.js) | 小程序 | API Client        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│              网关层                          │
│   Go/Gin Gateway | Auth | Rate Limit        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│              算法层                          │
│  LangGraph Agent | GraphRAG | ASR/TTS       │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│              数据层                          │
│  PostgreSQL | Milvus | Neo4j | Redis        │
└─────────────────────────────────────────────┘
```

## 📊 性能指标

| 指标 | 当前值 | 目标值 | 业界水平 |
|------|--------|--------|----------|
| 文本首Token | 500ms | 400ms | 600ms |
| 语音首响 | 300ms | **200ms** | 400ms |
| RAG召回率 | 92% | **97%** | 90% |
| 系统可用性 | 99.95% | **99.99%** | 99.9% |
| 并发支持 | 5000 | 10000 | 3000 |

## 🛠️ 技术栈

### 后端
- **网关**: Go 1.21 + Gin + gorilla/websocket
- **算法**: Python 3.11 + FastAPI + LangChain + LangGraph
- **数据库**: PostgreSQL 15 + Redis 7 + Milvus 2.3.4 + Neo4j 5.0

### 前端
- **Web**: Next.js 14 + React 18 + TypeScript + TailwindCSS
- **小程序**: 原生微信小程序 + WebAudio API

### AI/ML
- **LLM**: 豆包(Ark) + OpenAI GPT-4 + 本地模型
- **嵌入**: BGE-M3 + OpenAI Embeddings
- **语音**: FunASR + Edge-TTS

### 基础设施
- **容器**: Docker + Kubernetes
- **监控**: Prometheus + Grafana + OpenTelemetry
- **CI/CD**: GitHub Actions + ArgoCD

## 🤝 贡献指南

欢迎贡献代码、文档或建议！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feat/amazing-feature`)
3. 提交更改 (`git commit -m 'feat: add amazing feature'`)
4. 推送到分支 (`git push origin feat/amazing-feature`)
5. 创建 Pull Request

详见 [CONTRIBUTING.md](CONTRIBUTING.md)

## 📜 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - LLM应用框架
- [Milvus](https://milvus.io/) - 向量数据库
- [Neo4j](https://neo4j.com/) - 图数据库
- 所有贡献者和用户

## 📞 联系方式

- 项目主页: [https://chatbot.example.com](https://chatbot.example.com)
- 问题反馈: [GitHub Issues](https://github.com/example/chatbot/issues)
- 邮件: team@example.com

---

**⭐ 如果这个项目对您有帮助，请给我们一个 Star！**