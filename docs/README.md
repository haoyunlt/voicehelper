# VoiceHelper 文档中心

## 🎯 核心文档

### 📋 项目状态
- **[开发状态总览](dev-state.md)** - 项目当前状态和技术架构
- **[架构设计文档](ARCHITECTURE_DESIGN.md)** - 系统架构和技术决策  
- **[2026迭代计划](ITERATION_PLAN_2026.md)** - 18个月发展路线图

### 🛠️ 开发指南
- **[API使用指南](API_GUIDE.md)** - 完整API文档和SDK
- **[部署指南](DEPLOYMENT_GUIDE.md)** - 生产环境部署方案
- **[最佳实践指南](BEST_PRACTICES.md)** - 开发和运维最佳实践

### 🔧 运维支持  
- **[故障排除指南](TROUBLESHOOTING_GUIDE.md)** - 问题诊断和解决
- **[环境配置指南](ENVIRONMENT_GUIDE.md)** - 环境变量配置
- **[测试指南](TESTING_GUIDE.md)** - 测试策略和方法

## 🚀 快速开始

### 新用户
1. 阅读 [开发状态总览](dev-state.md) 了解项目
2. 查看 [架构设计文档](ARCHITECTURE_DESIGN.md) 理解系统
3. 参考 [API使用指南](API_GUIDE.md) 开始开发

### 开发者
1. [部署指南](DEPLOYMENT_GUIDE.md) - 环境搭建
2. [最佳实践指南](BEST_PRACTICES.md) - 开发规范  
3. [测试指南](TESTING_GUIDE.md) - 质量保障

### 运维人员
1. [部署指南](DEPLOYMENT_GUIDE.md) - 生产部署
2. [故障排除指南](TROUBLESHOOTING_GUIDE.md) - 问题解决
3. [环境配置指南](ENVIRONMENT_GUIDE.md) - 系统配置

## 📊 项目概览

**VoiceHelper** 是下一代多模态AI助手平台，支持语音、文本、图像的智能交互。

### 核心特性
- 🎤 **实时语音交互** - WebRTC + ASR/TTS
- 💬 **智能对话** - LLM + RAG检索增强  
- 🧠 **Agent编排** - LangGraph工作流
- 🌐 **多平台支持** - Web/Mobile/Desktop
- ☁️ **云原生架构** - K8s + 微服务

### 技术架构
```
客户端层: Web/Mobile/Desktop/Extension
    ↓
网关层: Go Gateway (HTTP/gRPC/WebSocket)  
    ↓
服务层: AI/Voice/Chat/RAG Services
    ↓
数据层: PostgreSQL/Redis/VectorDB/S3
```

### 当前状态
- **后端网关**: 95% 完成 ✅
- **AI算法服务**: 90% 完成 ✅  
- **前端应用**: 85% 完成 ✅
- **部署运维**: 92% 完成 ✅

## 📈 发展规划

### 2025年 Q4 - 技术债务清理
- 性能优化 (内存使用率降至70%)
- 安全加固 (SQL注入防护)
- 代码质量提升 (重复率降至3%)

### 2026年 - 功能增强  
- 实时语音革命 (OpenAI Realtime API)
- 多模态交互 (视觉理解)
- 智能Agent增强 (高级推理)
- 企业级功能 (多租户/SSO)

### 2027年 - 生态平台
- 开放API生态
- 插件市场
- 行业解决方案

## 🤝 参与贡献

### 文档贡献
```bash
git checkout -b docs/update-xxx
# 编辑文档
git commit -m "docs: update xxx"  
git push origin docs/update-xxx
```

### 获取帮助
- **技术支持**: support@voicehelper.com
- **GitHub**: [voicehelper/voicehelper](https://github.com/your-org/voicehelper)

---

*简洁、实用、易维护的文档体系*
