# VoiceHelper Kubernetes 部署

## 🚀 快速开始

在Docker Desktop的Kubernetes集群中一键部署VoiceHelper所有服务！

### 📋 前提条件

1. **Docker Desktop** 已安装并启用Kubernetes
2. **kubectl** 命令行工具可用
3. **至少 8GB 内存** 和 **50GB 磁盘空间**

### ⚡ 一键部署

```bash
# 1. 克隆项目
git clone https://github.com/voicehelper/voicehelper.git
cd voicehelper

# 2. 配置API密钥 (重要!)
cp env.unified .env
# 编辑 .env 文件，设置真实的 ARK_API_KEY 和 GLM_API_KEY

# 3. 一键部署
./deploy-k8s.sh deploy

# 4. 配置hosts文件
echo '127.0.0.1 voicehelper.local admin.voicehelper.local' | sudo tee -a /etc/hosts

# 5. 访问服务
open http://voicehelper.local
```

## 🎯 部署内容

### 核心服务 (voicehelper namespace)
- ✅ **API网关** - Go服务，端口8080
- ✅ **算法服务** - Python AI服务，端口8000  
- ✅ **语音服务** - 语音处理服务，端口8001
- ✅ **前端应用** - Next.js Web应用，端口3000
- ✅ **PostgreSQL** - 主数据库，端口5432
- ✅ **Redis** - 缓存服务，端口6379
- ✅ **Milvus** - 向量数据库，端口19530
- ✅ **Neo4j** - 图数据库，端口7687
- ✅ **NATS** - 消息队列，端口4222

### Dify AI平台 (voicehelper-dify namespace)
- ✅ **Dify API** - AI应用API服务，端口5001
- ✅ **Dify Web** - 可视化控制台，端口3000
- ✅ **Dify Worker** - 后台任务处理
- ✅ **Weaviate** - Dify向量数据库，端口8080
- ✅ **Dify PostgreSQL** - 独立数据库，端口5432
- ✅ **Dify Redis** - 独立缓存，端口6379
- ✅ **Sandbox** - 代码执行环境，端口8194

### 监控工具 (voicehelper-monitoring namespace)
- ✅ **Prometheus** - 指标收集，端口9090
- ✅ **Grafana** - 数据可视化，端口3000
- ✅ **Jaeger** - 分布式追踪，端口16686
- ✅ **pgAdmin** - 数据库管理，端口80
- ✅ **Redis Commander** - Redis管理，端口8081
- ✅ **Attu** - Milvus管理，端口3000

## 🌐 服务访问

### 主要服务
| 服务 | 地址 | 描述 |
|------|------|------|
| **VoiceHelper Web** | http://voicehelper.local | 主应用界面 |
| **Dify控制台** | http://voicehelper.local/dify | AI应用管理 |
| **API网关** | http://voicehelper.local/api | REST API |

### 管理工具
| 工具 | 地址 | 用户名 | 密码 |
|------|------|--------|------|
| **pgAdmin** | http://admin.voicehelper.local/pgadmin | admin@voicehelper.ai | admin123 |
| **Grafana** | http://admin.voicehelper.local/grafana | admin | admin123 |
| **Redis Commander** | http://admin.voicehelper.local/redis | - | - |
| **Prometheus** | http://admin.voicehelper.local/prometheus | - | - |

### 直接访问 (NodePort)
| 服务 | 地址 | 描述 |
|------|------|------|
| **HTTP入口** | http://localhost:30080 | 绕过域名直接访问 |
| **HTTPS入口** | https://localhost:30443 | SSL访问 |

## 🛠️ 管理命令

### 查看状态
```bash
# 查看所有服务状态
./deploy-k8s.sh status

# 查看特定命名空间
kubectl get pods -n voicehelper
kubectl get pods -n voicehelper-dify
kubectl get pods -n voicehelper-monitoring
```

### 查看日志
```bash
# 查看网关日志
./deploy-k8s.sh logs gateway

# 查看算法服务日志
kubectl logs -f deployment/algo-service -n voicehelper

# 查看Dify API日志
kubectl logs -f deployment/dify-api -n voicehelper-dify
```

### 扩缩容
```bash
# 扩容网关到5个副本
./deploy-k8s.sh scale gateway=5

# 扩容算法服务到3个副本
kubectl scale deployment algo-service --replicas=3 -n voicehelper
```

### 重启服务
```bash
# 重启所有服务
./deploy-k8s.sh restart

# 重启特定服务
kubectl rollout restart deployment/gateway -n voicehelper
```

## 🔧 部署选项

### 分组件部署
```bash
# 仅部署核心服务
./deploy-k8s.sh -c core deploy

# 仅部署Dify平台
./deploy-k8s.sh -c dify deploy

# 仅部署监控工具
./deploy-k8s.sh -c monitoring deploy
```

### 使用Helm部署
```bash
# 安装Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# 使用Helm部署
./deploy-k8s.sh -m helm deploy
```

### 自定义命名空间
```bash
# 部署到自定义命名空间
./deploy-k8s.sh -n my-voicehelper deploy
```

## 🔍 故障排除

### 常见问题

#### Pod启动失败
```bash
# 查看Pod详情
kubectl describe pod <pod-name> -n <namespace>

# 查看Pod日志
kubectl logs <pod-name> -n <namespace>
```

#### 服务无法访问
```bash
# 检查Service状态
kubectl get svc -n voicehelper

# 检查Ingress状态
kubectl get ingress -A

# 验证hosts文件
cat /etc/hosts | grep voicehelper
```

#### 存储问题
```bash
# 检查持久卷状态
kubectl get pv
kubectl get pvc -A

# 清理存储 (谨慎操作)
kubectl delete pvc --all -n voicehelper
```

### 完全清理
```bash
# 删除所有部署
./deploy-k8s.sh undeploy

# 强制删除 (包括数据)
./deploy-k8s.sh -f undeploy

# 清理持久卷
kubectl delete pv --all
```

## 📊 资源使用

### 最小配置
- **CPU**: 4核心
- **内存**: 8GB
- **存储**: 50GB

### 推荐配置
- **CPU**: 8核心
- **内存**: 16GB
- **存储**: 100GB SSD

### 实际使用情况
```bash
# 查看资源使用
kubectl top nodes
kubectl top pods -A

# 查看存储使用
kubectl get pvc -A
```

## 🔄 数据备份

### 自动备份
```bash
# 执行完整备份
./deploy-k8s.sh backup

# 备份位置
ls -la ./backups/k8s/
```

### 恢复数据
```bash
# 从备份恢复
./deploy-k8s.sh restore /path/to/backup
```

## 🚀 性能优化

### 启用自动扩缩容
```bash
# HPA已默认启用，查看状态
kubectl get hpa -A
```

### 调整资源限制
```bash
# 编辑部署配置
kubectl edit deployment gateway -n voicehelper

# 或使用patch命令
kubectl patch deployment gateway -n voicehelper -p '{"spec":{"template":{"spec":{"containers":[{"name":"gateway","resources":{"limits":{"memory":"4Gi","cpu":"2000m"}}}]}}}}'
```

## 📚 详细文档

- **[完整部署指南](./docs/KUBERNETES_DEPLOYMENT_GUIDE.md)** - 详细的Kubernetes部署文档
- **[Dify集成指南](./docs/DIFY_INTEGRATION_GUIDE.md)** - Dify AI平台集成说明
- **[故障排除指南](./docs/TROUBLESHOOTING_GUIDE.md)** - 常见问题解决方案

## 🆘 获取帮助

### 问题反馈
- **GitHub Issues**: https://github.com/voicehelper/voicehelper/issues
- **讨论区**: https://github.com/voicehelper/voicehelper/discussions

### 技术支持
```bash
# 生成诊断报告
./deploy-k8s.sh status > diagnosis.txt
kubectl get events -A --sort-by='.lastTimestamp' >> diagnosis.txt

# 提交Issue时请附上诊断报告
```

---

## 🎉 部署成功！

恭喜！您已成功在Kubernetes中部署了完整的VoiceHelper AI平台。

**下一步:**
1. 访问 http://voicehelper.local 开始使用
2. 访问 http://voicehelper.local/dify 创建AI应用
3. 查看监控面板了解系统状态
4. 阅读详细文档了解更多功能

**享受您的AI助手之旅！** 🚀
