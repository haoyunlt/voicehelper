# VoiceHelper AI - Docker Compose 部署指南

> 🚀 **一键启动完整的多模态AI助手平台**

## 快速开始

### 🎯 一键启动
```bash
# 克隆项目
git clone <repository-url>
cd voicehelper

# 一键启动
./quick-start.sh
```

### 🔧 手动启动
```bash
# 1. 复制环境配置
cp env.unified .env

# 2. 启动服务
./deploy.sh -e dev up -d

# 3. 访问应用
open http://localhost:3000
```

## 📋 系统要求

- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **内存**: 4GB+ 可用内存
- **存储**: 10GB+ 可用空间
- **网络**: 互联网连接

## 🏗️ 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    VoiceHelper AI                           │
├─────────────────────────────────────────────────────────────┤
│  🌐 前端层                                                   │
│  ├── Next.js Web App (3000)                               │
│  ├── Developer Portal (3002)                              │
│  └── Admin Panel (5001)                                   │
├─────────────────────────────────────────────────────────────┤
│  🚪 网关层                                                   │
│  ├── Go API Gateway (8080)                                │
│  └── Nginx/HAProxy (80/443)                               │
├─────────────────────────────────────────────────────────────┤
│  🤖 AI 服务层                                                │
│  ├── Algorithm Service (8000)                             │
│  └── Voice Service (8001)                                 │
├─────────────────────────────────────────────────────────────┤
│  🗄️ 数据层                                                   │
│  ├── PostgreSQL (5432)                                    │
│  ├── Redis (6379)                                         │
│  ├── Milvus Vector DB (19530)                             │
│  ├── Neo4j Graph DB (7474/7687)                           │
│  └── NATS Message Queue (4222)                            │
├─────────────────────────────────────────────────────────────┤
│  📊 监控层                                                   │
│  ├── Grafana (3004)                                       │
│  ├── Prometheus (9090)                                    │
│  └── Jaeger (16686)                                       │
└─────────────────────────────────────────────────────────────┘
```

## 🎮 部署模式

### 开发模式
```bash
./deploy.sh -e dev up -d
# 或
make dev
```
- ✅ 热重载支持
- ✅ 详细日志输出
- ✅ 开发工具集成

### 生产模式
```bash
./deploy.sh -e prod up -d
# 或
make prod
```
- ✅ 性能优化
- ✅ 多副本部署
- ✅ 安全加固

### 本地模式
```bash
./deploy.sh -e local up -d
# 或
make local
```
- ✅ 快速启动
- ✅ 简化配置

## 🔧 管理命令

### 基础操作
```bash
# 查看状态
./deploy.sh status

# 查看日志
./deploy.sh logs

# 健康检查
./deploy.sh health

# 重启服务
./deploy.sh restart

# 停止服务
./deploy.sh down
```

### 数据管理
```bash
# 备份数据
./deploy.sh backup

# 恢复数据
./deploy.sh restore /path/to/backup

# 清理资源
./deploy.sh -f clean
```

### 扩缩容
```bash
# 扩容算法服务
./deploy.sh scale algo-service=3

# 扩容网关服务
./deploy.sh scale gateway=2
```

## 🌐 访问地址

### 主要应用
- **Web 应用**: http://localhost:3000
- **API 网关**: http://localhost:8080
- **算法服务**: http://localhost:8000
- **管理后台**: http://localhost:5001

### 管理工具
- **Grafana**: http://localhost:3004 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **pgAdmin**: http://localhost:5050
- **Redis 管理**: http://localhost:8081

## 🔑 配置说明

### API 密钥配置
编辑 `.env` 文件：
```bash
# AI 模型 API 密钥
ARK_API_KEY=your-ark-api-key-here      # 豆包大模型
GLM_API_KEY=your-glm-api-key-here      # GLM-4

# 语音服务
AZURE_SPEECH_KEY=your-azure-speech-key-here
```

### 获取 API 密钥
- **豆包 API**: https://console.volcengine.com/
- **GLM-4 API**: https://open.bigmodel.cn/
- **Azure 语音**: https://portal.azure.com/

## 🔍 故障排除

### 常见问题

#### 端口冲突
```bash
# 查看端口占用
lsof -i :3000

# 停止冲突服务或修改端口
```

#### 内存不足
```bash
# 检查内存使用
free -h

# 清理 Docker 资源
docker system prune -f
```

#### 服务启动失败
```bash
# 查看详细日志
./deploy.sh logs -s <service-name>

# 检查配置
./test-config.sh
```

### 日志查看
```bash
# 查看所有日志
./deploy.sh logs

# 查看特定服务
./deploy.sh -s gateway logs

# 实时跟踪
./deploy.sh logs -f
```

## 📈 性能优化

### 资源配置
- **最低**: 4核/8GB/20GB
- **推荐**: 8核/16GB/50GB
- **生产**: 16核/32GB/100GB

### 扩容策略
```bash
# 水平扩容
./deploy.sh scale gateway=3
./deploy.sh scale algo-service=2

# 垂直扩容（修改配置文件）
```

## 🔒 安全配置

### 生产环境建议
1. 修改默认密码
2. 启用 HTTPS
3. 配置防火墙
4. 定期更新镜像
5. 监控异常访问

## 📝 更新维护

### 版本更新
```bash
# 1. 备份数据
./deploy.sh backup

# 2. 拉取更新
git pull origin main

# 3. 重新部署
./deploy.sh restart
```

### 定期维护
- **每周**: 备份数据、清理日志
- **每月**: 更新镜像、安全检查
- **每季度**: 性能评估、容量规划

## 📚 文档索引

- **[完整部署指南](./DEPLOYMENT.md)** - 详细的部署说明
- **[部署方案总结](./DEPLOYMENT_SUMMARY.md)** - 完整功能总结
- **[开发状态文档](./docs/dev-state.md)** - 项目状态概览
- **[API 使用指南](./docs/API_GUIDE.md)** - API 接口文档

## 🛠️ 开发工具

### 配置验证
```bash
# 验证配置文件
./test-config.sh

# 环境检查
./deploy.sh check
```

### 代码质量
```bash
# 代码检查
make lint

# 代码格式化
make format

# 运行测试
make test
```

## 📞 技术支持

### 获取帮助
- **项目文档**: [docs/](./docs/)
- **问题反馈**: GitHub Issues
- **技术讨论**: GitHub Discussions

### 联系方式
- **邮箱**: support@voicehelper.ai
- **官网**: https://voicehelper.ai

---

## 🎉 开始使用

```bash
# 一键启动
./quick-start.sh

# 访问应用
open http://localhost:3000
```

**享受您的 VoiceHelper AI 之旅！** 🚀
