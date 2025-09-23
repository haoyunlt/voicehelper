# VoiceHelper AI - Docker Compose 部署方案总结

## 🎉 部署方案完成状态

### ✅ 已完成的功能

#### 1. 完整的 Docker Compose 架构
- **主配置文件**: `docker-compose.yml` - 完整的生产级配置
- **开发环境**: `docker-compose.dev.yml` - 支持热重载和调试
- **生产环境**: `docker-compose.prod.yml` - 性能优化和安全配置
- **本地环境**: `docker-compose.local.yml` - 快速体验配置

#### 2. 优化的 Dockerfile
- **Go 网关服务**: 多阶段构建，最小化镜像大小
- **Python 算法服务**: 分层构建，支持缓存优化
- **Next.js 前端**: 生产级构建，性能优化
- **管理后台**: 轻量级 Flask 应用

#### 3. 完整的第三方软件支持
- **数据库**: PostgreSQL 15 (主数据库)
- **缓存**: Redis 7 (会话和缓存)
- **向量数据库**: Milvus 2.3.3 (AI 向量搜索)
- **图数据库**: Neo4j 5.15 (知识图谱)
- **消息队列**: NATS (事件驱动)
- **对象存储**: MinIO (Milvus 依赖)
- **协调服务**: etcd (Milvus 依赖)

#### 4. 监控和管理工具
- **指标监控**: Prometheus + Grafana
- **链路追踪**: Jaeger
- **数据库管理**: pgAdmin
- **Redis 管理**: Redis Commander
- **向量数据库管理**: Attu
- **API 文档**: Swagger UI
- **邮件测试**: Mailhog

#### 5. 负载均衡和反向代理
- **开发环境**: Nginx 反向代理
- **生产环境**: HAProxy 负载均衡
- **SSL 终止**: 支持 HTTPS 配置
- **健康检查**: 自动故障转移

#### 6. 部署和管理脚本
- **主部署脚本**: `deploy.sh` - 功能完整的部署工具
- **快速启动**: `quick-start.sh` - 一键启动向导
- **配置测试**: `test-config.sh` - 配置文件验证
- **Makefile**: 简化常用操作

#### 7. 服务发现和健康检查
- **内部服务发现**: Docker 网络自动解析
- **健康检查**: 所有服务都配置了健康检查
- **依赖管理**: 服务启动顺序和依赖关系
- **故障恢复**: 自动重启和故障转移

#### 8. 网络和安全配置
- **隔离网络**: 专用 Docker 网络
- **端口映射**: 合理的端口分配
- **安全配置**: 生产级安全设置
- **资源限制**: CPU 和内存限制

## 🚀 快速开始

### 一键启动
```bash
# 方式1: 使用快速启动脚本
./quick-start.sh

# 方式2: 使用 Makefile
make quick-start

# 方式3: 手动启动
cp env.unified .env
./deploy.sh -e dev up -d
```

### 环境选择
```bash
# 开发环境 (推荐用于开发和调试)
./deploy.sh -e dev up -d
make dev

# 生产环境 (推荐用于生产部署)
./deploy.sh -e prod up -d
make prod

# 本地环境 (推荐用于快速体验)
./deploy.sh -e local up -d
make local
```

### 服务配置
```bash
# 仅启动核心服务
./deploy.sh -p core up -d
make core

# 仅启动监控服务
./deploy.sh -p monitoring up -d
make monitoring

# 仅启动开发工具
./deploy.sh -p tools up -d
make tools
```

## 📊 服务架构

### 应用服务层
```
┌─────────────────────────────────────────────────────────────┐
│  🌐 Web Frontend (Next.js)     :3000                       │
│  🚪 API Gateway (Go)           :8080                       │
│  🤖 AI Algorithm Service (Python) :8000                    │
│  🎤 Voice Service (Python)     :8001                       │
│  ⚙️  Admin Panel (Flask)        :5001                       │
│  🔧 Developer Portal (Next.js) :3002                       │
└─────────────────────────────────────────────────────────────┘
```

### 数据存储层
```
┌─────────────────────────────────────────────────────────────┐
│  🗄️  PostgreSQL               :5432                         │
│  🔴 Redis                     :6379                         │
│  🔍 Milvus Vector DB          :19530                        │
│  📊 Neo4j Graph DB            :7474, :7687                  │
│  📨 NATS Message Queue        :4222                         │
└─────────────────────────────────────────────────────────────┘
```

### 监控和管理层
```
┌─────────────────────────────────────────────────────────────┐
│  📈 Grafana                   :3004                         │
│  📊 Prometheus                :9090                         │
│  🔍 Jaeger                    :16686                        │
│  🗄️  pgAdmin                   :5050                         │
│  🔴 Redis Commander           :8081                         │
│  🔍 Attu (Milvus UI)          :3001                         │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 管理命令

### 基础操作
```bash
# 查看服务状态
./deploy.sh status
make status

# 查看日志
./deploy.sh logs
make logs

# 健康检查
./deploy.sh health
make health

# 重启服务
./deploy.sh restart
make restart

# 停止服务
./deploy.sh down
make stop
```

### 数据管理
```bash
# 备份数据
./deploy.sh backup
make backup

# 恢复数据
./deploy.sh restore /path/to/backup
make restore BACKUP_DIR=/path/to/backup

# 清理资源
./deploy.sh -f clean
make clean
```

### 扩缩容
```bash
# 扩容算法服务
./deploy.sh scale algo-service=3
make scale-algo REPLICAS=3

# 扩容网关服务
./deploy.sh scale gateway=2
make scale-gateway REPLICAS=2
```

## 🔑 配置要点

### API 密钥配置
编辑 `.env` 文件，配置以下关键参数：
```bash
# AI 模型 API 密钥
ARK_API_KEY=your-ark-api-key-here      # 豆包大模型
GLM_API_KEY=your-glm-api-key-here      # GLM-4

# 语音服务 API 密钥
AZURE_SPEECH_KEY=your-azure-speech-key-here
AZURE_SPEECH_REGION=eastus

# 数据库配置
POSTGRES_PASSWORD=your-secure-password
REDIS_PASSWORD=your-secure-password

# JWT 安全配置
JWT_SECRET=your-jwt-secret-key
```

### 获取 API 密钥
1. **豆包 API**: https://console.volcengine.com/
2. **GLM-4 API**: https://open.bigmodel.cn/
3. **Azure 语音**: https://portal.azure.com/

## 🎯 访问地址

### 开发环境
- **Web 应用**: http://localhost:3000
- **API 网关**: http://localhost:8080
- **算法服务**: http://localhost:8000
- **语音服务**: http://localhost:8001
- **管理后台**: http://localhost:5001
- **开发者门户**: http://localhost:3002

### 管理工具
- **Grafana**: http://localhost:3004 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **pgAdmin**: http://localhost:5050 (admin@voicehelper.ai/admin123)
- **Redis 管理**: http://localhost:8081
- **Milvus 管理**: http://localhost:3001
- **API 文档**: http://localhost:8082
- **邮件测试**: http://localhost:8025

## 🔍 故障排除

### 常见问题解决

#### 1. 端口冲突
```bash
# 查看端口占用
lsof -i :3000
netstat -tulpn | grep :3000

# 修改端口配置或停止冲突服务
```

#### 2. 内存不足
```bash
# 检查内存使用
free -h
docker system df

# 清理资源
docker system prune -f
```

#### 3. 服务启动失败
```bash
# 查看详细日志
./deploy.sh logs -s <service-name>

# 重启特定服务
docker-compose restart <service-name>
```

#### 4. 配置文件错误
```bash
# 验证配置文件
./test-config.sh

# 检查环境变量
cat .env
```

### 日志查看
```bash
# 查看所有服务日志
./deploy.sh logs

# 查看特定服务日志
./deploy.sh -s gateway logs

# 实时跟踪日志
./deploy.sh logs -f
```

## 📈 性能优化

### 资源配置建议

#### 最低配置
- **CPU**: 4 核心
- **内存**: 8GB
- **存储**: 20GB SSD

#### 推荐配置
- **CPU**: 8 核心
- **内存**: 16GB
- **存储**: 50GB SSD

#### 生产配置
- **CPU**: 16 核心
- **内存**: 32GB
- **存储**: 100GB SSD

### 扩容策略
```bash
# 水平扩容
./deploy.sh scale gateway=3
./deploy.sh scale algo-service=2

# 垂直扩容 (修改 docker-compose.prod.yml)
resources:
  limits:
    memory: 4G
    cpus: '2'
```

## 🔒 安全配置

### 生产环境安全建议
1. **修改默认密码**
2. **启用 HTTPS**
3. **配置防火墙**
4. **定期更新镜像**
5. **监控异常访问**

### 网络安全
```bash
# 仅暴露必要端口
ports:
  - "80:80"    # HTTP
  - "443:443"  # HTTPS

# 使用内部网络通信
networks:
  - voicehelper-network
```

## 📝 更新和维护

### 版本更新流程
```bash
# 1. 备份数据
./deploy.sh backup

# 2. 拉取最新代码
git pull origin main

# 3. 更新镜像
./deploy.sh pull

# 4. 重新部署
./deploy.sh restart

# 5. 验证更新
./deploy.sh health
```

### 定期维护任务
- **每周**: 备份数据、清理日志
- **每月**: 更新镜像、检查安全
- **每季度**: 性能评估、容量规划

## 🎉 部署成功标志

当您看到以下输出时，说明部署成功：

```bash
✅ 所有服务运行正常
🌐 Web 应用: http://localhost:3000
🔧 API 网关: http://localhost:8080
🤖 算法服务: http://localhost:8000
📊 监控面板: http://localhost:3004
```

## 📞 技术支持

### 获取帮助
- **项目文档**: [DEPLOYMENT.md](./DEPLOYMENT.md)
- **配置指南**: [docs/](./docs/)
- **问题反馈**: GitHub Issues
- **技术讨论**: GitHub Discussions

### 联系方式
- **邮箱**: support@voicehelper.ai
- **官网**: https://voicehelper.ai
- **文档**: https://docs.voicehelper.ai

---

## 🏆 总结

VoiceHelper AI 的 Docker Compose 部署方案已经完成，提供了：

✅ **完整的服务架构** - 包含所有必要的应用服务和第三方依赖
✅ **多环境支持** - 开发、生产、本地三种环境配置
✅ **一键部署** - 简化的启动流程和管理工具
✅ **完善的监控** - 全面的监控和管理界面
✅ **生产就绪** - 性能优化和安全配置
✅ **易于维护** - 完整的文档和故障排除指南

现在您可以使用 `./quick-start.sh` 一键启动完整的 VoiceHelper AI 系统！

🎯 **立即开始**: `./quick-start.sh`
