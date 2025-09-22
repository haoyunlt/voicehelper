# VoiceHelper 快速入门指南

## 🚀 5分钟快速体验

### 前置条件

确保你的系统已安装：
- Docker >= 20.10
- Docker Compose >= 2.0

### 一键启动

```bash
# 1. 克隆项目
git clone https://github.com/your-org/voicehelper.git
cd voicehelper

# 2. 启动所有服务
cd deploy
docker-compose up -d

# 3. 等待服务启动（约2-3分钟）
echo "等待服务启动中..."
sleep 180

# 4. 验证服务状态
curl -s http://localhost:8080/health | jq '.'
curl -s http://localhost:8000/health | jq '.'
```

### 立即体验

**Web 界面**: 打开浏览器访问 http://localhost:3000

**API 测试**:
```bash
# 发送第一条消息
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好，我想了解VoiceHelper的功能",
    "user_id": "quickstart_user"
  }' | jq '.'
```

## 📱 核心功能演示

### 1. 智能对话

**Web 界面体验**:
1. 访问 http://localhost:3000/chat
2. 在输入框中输入："你好，请介绍一下你的功能"
3. 点击发送，查看AI回复

**API 体验**:
```bash
# 创建对话
CONV_ID=$(curl -s -X POST http://localhost:8080/api/v1/conversations \
  -H "Content-Type: application/json" \
  -d '{"user_id": "demo_user", "channel": "api"}' | jq -r '.conversation_id')

# 发送消息
curl -X POST "http://localhost:8080/api/v1/conversations/$CONV_ID/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "请帮我解释一下人工智能的基本概念",
    "message_type": "text"
  }' | jq '.reply'
```

### 2. 语音功能

**语音识别测试**:
```bash
# 准备测试音频文件（或使用示例文件）
curl -X POST http://localhost:8000/api/v1/voice/asr \
  -F "audio=@tests/datasets/voice/sample.wav" \
  -F "language=zh-CN" | jq '.'
```

**语音合成测试**:
```bash
# 文本转语音
curl -X POST http://localhost:8000/api/v1/voice/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "欢迎使用VoiceHelper智能助手",
    "voice": "zh-CN-XiaoxiaoNeural",
    "speed": 1.0
  }' --output welcome.wav

# 播放生成的音频
# macOS: afplay welcome.wav
# Linux: aplay welcome.wav
```

### 3. 知识库问答

**上传测试文档**:
```bash
# 创建测试文档
echo "VoiceHelper是一个智能聊天机器人平台，支持语音交互和知识问答。主要功能包括：
1. 自然语言对话
2. 语音识别和合成
3. 知识库检索
4. 多渠道接入
5. 数据分析" > test_doc.txt

# 上传到知识库
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@test_doc.txt" \
  -F "title=VoiceHelper功能介绍" \
  -F "category=product_info" | jq '.'
```

**知识检索测试**:
```bash
# 等待文档处理完成（约30秒）
sleep 30

# 搜索知识
curl -X POST http://localhost:8000/api/v1/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "VoiceHelper有哪些主要功能？",
    "top_k": 3
  }' | jq '.results'

# RAG问答
curl -X POST http://localhost:8000/api/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "请详细介绍VoiceHelper的语音功能",
    "top_k": 5
  }' | jq '.answer'
```

### 4. 数据分析

**访问分析面板**:
1. 打开 http://localhost:3000/analytics
2. 查看对话统计、用户分析等数据

**API 获取统计数据**:
```bash
# 获取今日对话统计
curl -s "http://localhost:8080/api/v1/analytics/conversations?period=1d" | jq '.'

# 获取用户活跃度
curl -s "http://localhost:8080/api/v1/analytics/users?period=7d" | jq '.'

# 获取系统性能指标
curl -s "http://localhost:8080/api/v1/analytics/performance" | jq '.'
```

## 🔧 开发环境设置

### 本地开发

**1. 启动基础服务**:
```bash
# 只启动数据库等基础服务
cd deploy
```

**2. 启动后端服务**:
```bash
cd backend
go mod tidy
go run cmd/server/main.go
```

**3. 启动算法服务**:
```bash
cd algo
pip install -r requirements.txt
python app/main.py
```

**4. 启动前端服务**:
```bash
cd frontend
npm install
npm run dev
```

### 环境变量配置

**后端环境变量** (backend/.env):
```bash
PORT=8080
GIN_MODE=debug
DATABASE_URL=postgresql://postgres:password@localhost:5432/voicehelper
REDIS_URL=redis://localhost:6379
LOG_LEVEL=debug
```

**算法服务环境变量** (algo/.env):
```bash
HOST=0.0.0.0
PORT=8000
MILVUS_HOST=localhost
MILVUS_PORT=19530
OPENAI_API_KEY=your-openai-api-key
```

**前端环境变量** (frontend/.env.local):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8080
NEXT_PUBLIC_WS_URL=ws://localhost:8080
NEXT_PUBLIC_ENABLE_VOICE=true
```

## 🧪 运行测试

### E2E 测试

```bash
# 安装测试依赖
cd tests/e2e
npm install

# 运行冒烟测试
npm run test:smoke

# 运行完整测试套件
npm test
```

### 模块测试

```bash
# 运行模块测试
python3 tests/module_test_runner.py

# 查看测试报告
cat tests/MODULE_TEST_REPORT.md
```

### 性能测试

```bash
# 运行性能测试
cd tests/performance
python3 comprehensive_performance_test.py

# 查看性能报告
cat performance_report.json
```

## 📊 监控和日志

### 查看服务日志

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f backend
docker-compose logs -f algo-service
docker-compose logs -f frontend

# 查看最近的错误日志
docker-compose logs --tail=100 backend | grep ERROR
```

### 监控服务状态

```bash
# 检查服务健康状态
curl http://localhost:8080/health
curl http://localhost:8000/health

# 查看容器状态
docker-compose ps

# 查看资源使用情况
docker stats
```

### 性能监控

```bash
# 启动监控服务（可选）
docker-compose -f docker-compose.monitoring.yml up -d

# 访问监控面板
# Grafana: http://localhost:3001 (admin/admin)
# Prometheus: http://localhost:9090
```

## 🔍 故障排除

### 常见问题

**1. 服务启动失败**
```bash
# 检查端口占用
lsof -i :8080
lsof -i :8000
lsof -i :3000

# 清理并重启
docker-compose down
docker system prune -f
docker-compose up -d
```

**2. 数据库连接问题**
```bash
# 检查数据库状态
docker-compose exec postgres pg_isready

# 重置数据库
docker-compose down postgres
docker volume rm deploy_postgres_data
docker-compose up -d postgres
```

**3. 向量数据库问题**
```bash
curl http://localhost:19530/health

```

**4. 前端页面无法访问**
```bash
# 重新构建前端
docker-compose build --no-cache frontend
docker-compose up -d frontend

# 检查前端日志
docker-compose logs frontend
```

### 获取帮助

如果遇到问题，可以：

1. **查看日志**: `docker-compose logs service-name`
2. **检查文档**: 查看 `docs/` 目录下的详细文档
3. **运行诊断**: `./scripts/health-check.sh`
4. **提交问题**: 在 GitHub Issues 中描述问题

## 🎯 下一步

恭喜！你已经成功启动了 VoiceHelper 系统。接下来你可以：

### 1. 深入了解功能
- 📖 阅读 [完整使用指南](USER_GUIDE.md)
- 🔌 查看 [API 文档](api/)
- 🏗️ 了解 [系统架构](ARCHITECTURE_DEEP_DIVE.md)

### 2. 自定义配置
- ⚙️ 修改 [环境配置](ENVIRONMENT_CONFIG.md)
- 🎨 定制前端界面
- 🤖 调整 AI 模型参数

### 3. 集成到你的项目
- 🔗 使用 REST API
- 📦 集成 JavaScript/Python SDK
- 🌐 嵌入聊天组件

### 4. 生产部署
- 🚀 查看 [部署指南](DEPLOYMENT_GUIDE.md)
- 🔒 配置安全设置
- 📊 设置监控告警

## 📞 技术支持

- **文档**: [docs/](.)
- **示例**: [examples/](../examples/)
- **问题反馈**: [GitHub Issues](https://github.com/your-org/voicehelper/issues)
- **社区讨论**: [Discussions](https://github.com/your-org/voicehelper/discussions)

---

**快速入门完成！** 🎉

现在你已经掌握了 VoiceHelper 的基本使用方法，可以开始构建你的智能对话应用了！
