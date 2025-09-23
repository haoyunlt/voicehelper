# VoiceHelper + Dify AI 平台集成

## 🎉 集成完成

VoiceHelper项目已成功集成Dify AI平台，提供强大的可视化AI应用构建能力！

## 🚀 快速开始

### 一键启动

```bash
# 克隆项目
git clone https://github.com/voicehelper/voicehelper.git
cd voicehelper

# 配置API密钥
cp env.unified .env
# 编辑 .env 文件，设置 ARK_API_KEY 和 GLM_API_KEY

# 一键启动
./start-dify.sh
```

### 访问服务

- **VoiceHelper Web应用**: http://localhost:3000
- **Dify AI控制台**: http://localhost:3001 
- **集成API接口**: http://localhost:8200
- **数据库管理**: http://localhost:5050

## 📋 主要功能

### ✅ 已实现功能

1. **完整的Dify AI平台部署**
   - Dify Web控制台 (端口3001)
   - Dify API服务 (端口5001)  
   - Weaviate向量数据库 (端口8080)
   - 代码执行沙箱 (端口8194)

2. **VoiceHelper-Dify集成适配器**
   - 统一API接口 (端口8200)
   - 双向数据同步
   - 聊天和工作流集成
   - 知识库同步

3. **独立数据存储**
   - Dify专用PostgreSQL (端口5433)
   - Dify专用Redis (端口6380)
   - 与VoiceHelper数据隔离

4. **完善的管理工具**
   - Dify数据库管理 (端口5051)
   - Dify Redis管理 (端口8083)
   - 服务监控和日志

5. **灵活的部署选项**
   - 一键启动脚本 (`./start-dify.sh`)
   - 模块化部署 (`./deploy.sh -p dify`)
   - Docker Compose集成

## 🔧 配置文件

### 新增文件

```
├── docker-compose.dify.yml          # Dify服务配置
├── start-dify.sh                    # 一键启动脚本
├── integrations/
│   └── dify-adapter/               # 集成适配器
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── main.py
│       └── start.sh
├── tools/deployment/dify/          # Dify部署配置
│   ├── init-db.sql
│   └── pgadmin-servers.json
└── docs/
    ├── DIFY_INTEGRATION_GUIDE.md   # 详细集成指南
    └── DEPLOYMENT_GUIDE.md         # 更新的部署指南
```

### 更新文件

```
├── docker-compose.yml              # 添加Dify注释
├── deploy.sh                       # 支持Dify配置
├── env.unified                     # 添加Dify环境变量
└── docs/DEPLOYMENT_GUIDE.md        # 添加Dify部署章节
```

## 🎯 使用场景

### 1. AI应用快速构建

通过Dify可视化界面构建复杂AI工作流：

```bash
# 访问Dify控制台
open http://localhost:3001

# 创建AI应用
# 配置模型和提示词
# 发布应用获取API Key
```

### 2. 统一API调用

通过集成适配器统一调用：

```bash
# 聊天接口
curl -X POST http://localhost:8200/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "你好，请介绍VoiceHelper项目",
    "app_id": "your-dify-app-id"
  }'

# 工作流接口  
curl -X POST http://localhost:8200/api/v1/workflow/run \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_id": "your-workflow-id",
    "inputs": {"query": "分析文档"}
  }'
```

### 3. 知识库管理

- 在Dify中上传和管理文档
- 自动向量化和索引
- 与VoiceHelper知识库同步

## 📊 架构优势

### 🔄 双向集成

- **VoiceHelper → Dify**: 利用Dify的AI能力
- **Dify → VoiceHelper**: 使用VoiceHelper的语音和多模态能力

### 🗄️ 数据隔离

- Dify使用独立的数据库实例
- 避免数据冲突和性能影响
- 支持独立备份和恢复

### 🚀 性能优化

- 服务间通过内部网络通信
- 支持水平扩展
- 缓存和负载均衡

### 🔒 安全设计

- 服务间认证和授权
- 数据传输加密
- 访问控制和审计

## 📚 文档资源

- **[Dify集成指南](./docs/DIFY_INTEGRATION_GUIDE.md)** - 详细的集成和使用指南
- **[部署指南](./docs/DEPLOYMENT_GUIDE.md)** - 完整的部署文档
- **[API文档](http://localhost:8200/docs)** - 集成适配器API文档
- **[Dify官方文档](https://docs.dify.ai/)** - Dify平台官方文档

## 🛠️ 故障排除

### 常见问题

1. **端口冲突**
   ```bash
   # 检查端口占用
   lsof -i :3001
   lsof -i :5001
   
   # 修改端口配置
   # 编辑 env.unified 中的端口设置
   ```

2. **服务启动失败**
   ```bash
   # 查看日志
   docker logs voicehelper-dify-api
   
   # 重启服务
   ./deploy.sh -p dify restart
   ```

3. **内存不足**
   ```bash
   # 检查内存使用
   docker stats
   
   # 调整资源限制
   # 编辑 docker-compose.dify.yml
   ```

### 获取帮助

```bash
# 显示帮助信息
./start-dify.sh --help
./deploy.sh --help

# 检查服务状态
./deploy.sh status

# 查看服务日志
./deploy.sh logs
```

## 🎊 下一步

1. **创建第一个Dify应用**
   - 访问 http://localhost:3001
   - 注册账号并登录
   - 创建聊天助手应用

2. **配置模型提供商**
   - 在Dify中配置豆包、GLM等模型
   - 测试模型连接

3. **构建工作流**
   - 使用Dify的工作流编辑器
   - 集成VoiceHelper的语音能力

4. **API集成**
   - 获取应用API Key
   - 通过集成适配器调用

## 🤝 贡献

欢迎提交Issue和Pull Request来改进Dify集成功能！

---

**🎉 恭喜！VoiceHelper + Dify AI平台集成部署完成！**

现在您可以享受强大的可视化AI应用构建体验了！
