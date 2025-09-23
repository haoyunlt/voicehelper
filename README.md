# 🤖 VoiceHelper AI - 智能语音助手平台

[![Version](https://img.shields.io/badge/version-v3.0.0-blue.svg)](https://github.com/voicehelper/voicehelper)
[![Status](https://img.shields.io/badge/status-production_ready-green.svg)](https://github.com/voicehelper/voicehelper)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

🚀 **企业级智能语音助手平台，支持多模态交互、GraphRAG检索、智能Agent系统**

📚 **文档中心**: 查看 [docs/README.md](docs/README.md) 获取完整文档导航

---

## 🌟 核心特性

### 🎭 多模态交互
- **文本对话**: SSE流式输出，支持Markdown渲染
- **语音对话**: WebSocket全双工，支持实时语音交互
- **模态切换**: 同会话无缝切换，上下文完整保持

### 🧠 GraphRAG检索系统
- **知识图谱**: 实体抽取+关系构建，智能知识管理
- **多跳推理**: 图遍历推理，路径解释，社区发现
- **融合排序**: 多路召回，智能重排，高召回率

### 🤖 智能Agent系统
- **多推理模式**: 演绎/归纳/溯因/类比推理
- **规划能力**: 层次化任务分解，依赖管理
- **工具生态**: MCP协议，丰富的服务集成
- **记忆系统**: 多层次记忆管理

### 🌐 全平台支持
- **Web应用**: Next.js前端，响应式设计
- **API服务**: RESTful API，完整的开发者文档
- **多平台部署**: Docker容器化，Kubernetes支持

---

## 🚀 快速开始

### 环境要求
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **内存**: 8GB+ (推荐16GB)
- **存储**: 20GB可用空间

### 一键部署

```bash
# 1. 克隆项目
git clone https://github.com/voicehelper/voicehelper.git
cd voicehelper

# 2. 配置环境变量
cp env.unified .env
# 编辑 .env 文件，配置API密钥

# 3. 启动服务
./quick-start.sh

# 或使用Docker Compose
docker-compose -f docker-compose.local.yml up -d
```

### 服务访问

| 服务 | 地址 | 说明 |
|------|------|------|
| **Web界面** | http://localhost:3000 | 主应用界面 |
| **API网关** | http://localhost:8080 | 后端API服务 |
| **API文档** | http://localhost:8000/docs | Swagger文档 |
| **管理后台** | http://localhost:5001 | 系统管理 |

### 开发模式

```bash
# 安装依赖
make install

# 启动开发服务器
make dev

# 运行测试
make test

# 构建项目
make build
```

---

## 🏗️ 系统架构

### 技术栈
- **前端**: Next.js + React + TypeScript
- **后端**: Go + Python (FastAPI)
- **数据库**: PostgreSQL + Redis + Neo4j
- **AI模型**: 支持多种LLM (OpenAI, 豆包, GLM等)
- **部署**: Docker + Kubernetes

### 服务架构
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   前端应用   │───▶│   API网关   │───▶│  算法服务   │
│  (Next.js)  │    │    (Go)     │    │ (Python)   │
└─────────────┘    └─────────────┘    └─────────────┘
                           │
                           ▼
                   ┌─────────────┐
                   │   数据层    │
                   │ PG+Redis+Neo4j │
                   └─────────────┘
```

## 📚 文档导航

- **[快速开始](docs/QUICK_START.md)** - 详细的部署和使用指南
- **[开发指南](docs/DEVELOPMENT_GUIDE.md)** - 开发环境搭建和API使用
- **[部署指南](docs/DEPLOYMENT_GUIDE.md)** - 生产环境部署
- **[架构设计](docs/ARCHITECTURE_DESIGN.md)** - 系统架构和技术选型
- **[API文档](docs/API_GUIDE.md)** - 完整的API接口文档

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📞 联系方式

- **项目主页**: https://github.com/voicehelper/voicehelper
- **问题反馈**: https://github.com/voicehelper/voicehelper/issues
- **技术文档**: [docs/](docs/)

## 📜 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

⭐ **如果这个项目对您有帮助，请给我们一个 Star！**
