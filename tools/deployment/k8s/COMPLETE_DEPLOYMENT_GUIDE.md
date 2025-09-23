# VoiceHelper Kubernetes 完整部署指南

## 概述

本指南提供了VoiceHelper AI语音助手系统在Kubernetes上的完整部署方案，包括所有服务模块和第三方软件的本地部署，实现了高可用、可扩展、可观测的生产级部署。

## 架构特点

### ✅ 完整性
- **不简化功能**：保留所有原有功能和特性
- **全模块覆盖**：包含前端、后端、AI/ML、存储、监控等所有模块
- **完整的第三方软件**：PostgreSQL、Redis、MinIO、监控栈等全部本地部署

### ✅ 高可用性
- **多副本部署**：关键服务多副本运行
- **故障转移**：自动故障检测和转移
- **健康检查**：完善的存活性和就绪性探针
- **Pod中断预算**：保证服务连续性

### ✅ 可扩展性
- **自动扩缩容**：基于CPU、内存和自定义指标
- **水平扩展**：支持服务实例动态增减
- **资源优化**：合理的资源请求和限制

### ✅ 可观测性
- **完整监控**：Prometheus + Grafana + AlertManager
- **链路追踪**：Jaeger分布式追踪
- **日志聚合**：Fluentd + Elasticsearch
- **自定义指标**：业务指标监控

### ✅ 安全性
- **RBAC权限控制**：细粒度权限管理
- **网络策略**：微分段网络隔离
- **密钥管理**：Kubernetes Secrets
- **Pod安全策略**：容器安全限制

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            VoiceHelper K8s 架构                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Ingress   │  │   Gateway   │  │  Frontend   │  │ Admin Portal│        │
│  │  (Nginx)    │  │   (Go)      │  │  (Next.js)  │  │   (React)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                 │                 │                 │             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Voice Service│  │ Algo Service│  │ BGE Service │  │FAISS Service│        │
│  │ (WebRTC)    │  │  (Python)   │  │ (Embedding) │  │  (Search)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                 │                 │                 │             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ PostgreSQL  │  │Redis Cluster│  │   MinIO     │  │   Neo4j     │        │
│  │  (Master+   │  │ (6 nodes)   │  │ (4 nodes)   │  │ (3 nodes)   │        │
│  │  2 Replicas)│  │             │  │             │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                 │                 │                 │             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Prometheus  │  │   Grafana   │  │   Jaeger    │  │  Fluentd    │        │
│  │ (Monitoring)│  │(Dashboards) │  │  (Tracing)  │  │ (Logging)   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 部署文件结构

```
tools/deployment/k8s/
├── 00-prerequisites/           # 前置条件
│   ├── namespace.yaml         # 命名空间和资源配额
│   ├── rbac.yaml             # RBAC权限控制
│   └── storage-classes.yaml  # 存储类和持久化卷
├── 01-infrastructure/         # 基础设施
│   ├── configmaps.yaml       # 配置映射
│   └── secrets.yaml          # 密钥管理
├── 02-storage-services/       # 存储服务
│   ├── postgresql-cluster.yaml  # PostgreSQL高可用集群
│   ├── redis-cluster.yaml       # Redis集群
│   ├── minio-cluster.yaml       # MinIO对象存储
│   ├── neo4j-cluster.yaml       # Neo4j图数据库
│   └── elasticsearch-cluster.yaml # Elasticsearch日志存储
├── 03-messaging-services/     # 消息队列
│   ├── nats-jetstream.yaml    # NATS消息队列
│   ├── kafka-cluster.yaml     # Kafka事件流
│   └── rabbitmq-cluster.yaml  # RabbitMQ任务队列
├── 04-ai-ml-services/        # AI/ML服务
│   ├── bge-service.yaml      # BGE向量化服务
│   ├── faiss-service.yaml    # FAISS向量搜索
│   ├── llm-proxy.yaml        # LLM代理服务
│   ├── speech-service.yaml   # 语音识别服务
│   └── tts-service.yaml      # 语音合成服务
├── 05-application-services/   # 应用服务
│   ├── gateway.yaml          # API网关
│   ├── algo-service.yaml     # 算法服务
│   ├── voice-service.yaml    # 语音服务
│   ├── frontend.yaml         # 前端服务
│   └── admin-portal.yaml     # 管理门户
├── 06-monitoring-stack/      # 监控栈
│   ├── prometheus.yaml       # Prometheus监控
│   ├── grafana.yaml          # Grafana仪表盘
│   ├── alertmanager.yaml     # 告警管理
│   ├── jaeger.yaml           # 链路追踪
│   └── fluentd.yaml          # 日志收集
├── 07-service-mesh/          # 服务网格
│   ├── istio-base.yaml       # Istio基础组件
│   ├── istio-control-plane.yaml # Istio控制平面
│   └── virtual-services.yaml    # 虚拟服务配置
├── 08-ingress-networking/    # 网络入口
│   ├── nginx-ingress.yaml    # Nginx Ingress
│   ├── cert-manager.yaml     # 证书管理
│   └── ingress-rules.yaml    # Ingress规则
├── deploy-complete.sh        # 完整部署脚本
├── test-deployment.sh        # 部署测试脚本
└── COMPLETE_DEPLOYMENT_GUIDE.md # 本文档
```

## 资源需求

### 最小配置
- **节点数量**: 3个节点（1 Master + 2 Worker）
- **每节点配置**: 8C16G + 200GB SSD
- **总资源**: 24C48G + 600GB存储

### 推荐配置
- **节点数量**: 9个节点（3 Master + 6 Worker）
- **Master节点**: 4C8G + 100GB SSD
- **Worker节点**: 16C32G + 500GB SSD
- **总资源**: 108C208G + 3.3TB存储

### 生产配置
- **节点数量**: 12个节点（3 Master + 9 Worker）
- **Master节点**: 8C16G + 200GB SSD
- **Worker节点**: 32C64G + 1TB NVMe SSD
- **总资源**: 312C624G + 9.6TB存储

## 部署步骤

### 1. 环境准备

#### 1.1 安装依赖工具
```bash
# 安装kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# 安装Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# 安装Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

#### 1.2 验证Kubernetes集群
```bash
kubectl cluster-info
kubectl get nodes
kubectl get namespaces
```

#### 1.3 配置存储
```bash
# 创建本地存储目录（在每个节点上）
sudo mkdir -p /data/{postgres,redis,minio,faiss,prometheus,elasticsearch}
sudo chown -R 1000:1000 /data
```

### 2. 快速部署

#### 2.1 完整部署
```bash
# 克隆项目
git clone <repository-url>
cd voicehelper

# 执行完整部署
./tools/deployment/k8s/deploy-complete.sh full
```

#### 2.2 最小部署
```bash
# 仅部署核心服务
./tools/deployment/k8s/deploy-complete.sh minimal
```

#### 2.3 分步部署
```bash
# 仅部署存储服务
./tools/deployment/k8s/deploy-complete.sh storage-only

# 仅部署AI/ML服务
./tools/deployment/k8s/deploy-complete.sh ai-only

# 仅部署监控系统
./tools/deployment/k8s/deploy-complete.sh monitoring-only
```

### 3. 验证部署

#### 3.1 运行自动化测试
```bash
./tools/deployment/k8s/test-deployment.sh
```

#### 3.2 手动验证
```bash
# 检查所有Pod状态
kubectl get pods -A

# 检查服务状态
kubectl get services -A

# 检查Ingress状态
kubectl get ingress -A

# 检查持久化卷状态
kubectl get pv,pvc -A
```

### 4. 访问应用

#### 4.1 配置本地hosts
```bash
# 添加到 /etc/hosts
127.0.0.1 voicehelper.local
127.0.0.1 api.voicehelper.local
127.0.0.1 bge.voicehelper.local
127.0.0.1 faiss.voicehelper.local
127.0.0.1 grafana.voicehelper.local
127.0.0.1 prometheus.voicehelper.local
127.0.0.1 jaeger.voicehelper.local
```

#### 4.2 端口转发访问
```bash
# Gateway API
kubectl port-forward -n voicehelper service/gateway 8080:8080

# BGE服务
kubectl port-forward -n voicehelper service/bge-service 8080:8080

# FAISS服务
kubectl port-forward -n voicehelper service/faiss-client 8081:8081

# Grafana监控
kubectl port-forward -n voicehelper-monitoring service/grafana 3000:3000

# Prometheus
kubectl port-forward -n voicehelper-monitoring service/prometheus 9090:9090
```

#### 4.3 访问地址
- **主应用**: http://voicehelper.local
- **API网关**: http://api.voicehelper.local
- **BGE服务**: http://bge.voicehelper.local
- **FAISS服务**: http://faiss.voicehelper.local
- **Grafana**: http://grafana.voicehelper.local (admin/VoiceHelper2025!)
- **Prometheus**: http://prometheus.voicehelper.local

## 监控和运维

### 监控指标

#### 应用指标
- **QPS/TPS**: 每秒请求/事务数
- **响应时间**: P50/P95/P99延迟
- **错误率**: 4xx/5xx错误比例
- **业务指标**: 向量化速度、搜索准确率

#### 基础设施指标
- **CPU使用率**: 节点和Pod级别
- **内存使用率**: 包括缓存和缓冲区
- **磁盘I/O**: 读写速度和IOPS
- **网络流量**: 入站/出站流量

#### 自定义指标
- **AI模型推理时间**: BGE向量化延迟
- **向量搜索延迟**: FAISS搜索性能
- **语音处理质量**: ASR/TTS准确率
- **用户体验指标**: 端到端响应时间

### 告警策略

#### 关键告警（Critical）
- 服务不可用（>1分钟）
- 错误率超过5%（>5分钟）
- 响应时间超过阈值（>5分钟）
- 磁盘空间不足（<10%）

#### 警告告警（Warning）
- CPU使用率超过80%（>5分钟）
- 内存使用率超过85%（>5分钟）
- Pod重启频繁（>5次/小时）
- 数据库连接数过高（>80%）

### 扩缩容策略

#### 自动扩缩容
```yaml
# HPA配置示例
minReplicas: 2
maxReplicas: 10
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 70
- type: Resource
  resource:
    name: memory
    target:
      type: Utilization
      averageUtilization: 80
```

#### 手动扩缩容
```bash
# 扩展Gateway服务
kubectl scale deployment gateway -n voicehelper --replicas=5

# 扩展BGE服务
kubectl scale deployment bge-service -n voicehelper --replicas=4

# 扩展FAISS服务
kubectl scale statefulset faiss-service -n voicehelper --replicas=3
```

## 备份和恢复

### 数据备份

#### 自动备份
- **PostgreSQL**: 每日全量备份 + 连续WAL备份
- **Redis**: 每小时RDB快照 + AOF日志
- **MinIO**: 每日对象同步备份
- **FAISS**: 每日索引文件备份

#### 手动备份
```bash
# PostgreSQL备份
kubectl exec -n voicehelper-storage postgres-master-0 -- pg_dump -U voicehelper voicehelper > backup.sql

# Redis备份
kubectl exec -n voicehelper-storage redis-cluster-0 -- redis-cli BGSAVE

# MinIO备份
kubectl exec -n voicehelper-storage minio-0 -- mc mirror local/voicehelper-data backup/
```

### 灾难恢复

#### RTO/RPO目标
- **RTO**: 15分钟（恢复时间目标）
- **RPO**: 5分钟（恢复点目标）

#### 恢复流程
1. **评估故障范围**：确定受影响的服务和数据
2. **启动应急响应**：通知相关人员，启动应急流程
3. **数据恢复**：从最近的备份恢复数据
4. **服务重启**：重新部署受影响的服务
5. **验证功能**：确保所有功能正常工作
6. **恢复流量**：逐步恢复用户流量

## 安全配置

### 网络安全
- **网络策略**: 微分段网络隔离
- **服务网格**: Istio mTLS加密
- **Ingress安全**: SSL/TLS终止
- **防火墙规则**: 最小权限原则

### 访问控制
- **RBAC**: 基于角色的访问控制
- **ServiceAccount**: 服务账户隔离
- **Pod安全策略**: 容器安全限制
- **密钥管理**: Kubernetes Secrets

### 合规性
- **数据加密**: 传输和存储加密
- **审计日志**: 完整的操作审计
- **数据脱敏**: PII数据保护
- **合规检查**: 定期安全扫描

## 故障排查

### 常见问题

#### Pod无法启动
```bash
# 查看Pod状态
kubectl describe pod <pod-name> -n <namespace>

# 查看Pod日志
kubectl logs <pod-name> -n <namespace>

# 查看事件
kubectl get events -n <namespace> --sort-by='.lastTimestamp'
```

#### 服务无法访问
```bash
# 检查服务配置
kubectl describe service <service-name> -n <namespace>

# 检查端点
kubectl get endpoints <service-name> -n <namespace>

# 测试网络连通性
kubectl run test-pod --rm -i --restart=Never --image=curlimages/curl -- curl <service-url>
```

#### 存储问题
```bash
# 检查PVC状态
kubectl describe pvc <pvc-name> -n <namespace>

# 检查存储类
kubectl describe storageclass <storage-class-name>

# 检查持久化卷
kubectl describe pv <pv-name>
```

### 日志查看
```bash
# 查看应用日志
kubectl logs -f deployment/gateway -n voicehelper

# 查看系统日志
kubectl logs -f daemonset/fluentd -n voicehelper-monitoring

# 查看监控日志
kubectl logs -f deployment/prometheus -n voicehelper-monitoring
```

## 性能优化

### 资源优化
- **CPU请求/限制**: 根据实际使用情况调整
- **内存请求/限制**: 避免OOM和资源浪费
- **存储优化**: 选择合适的存储类型
- **网络优化**: 使用服务网格优化通信

### 应用优化
- **连接池**: 数据库和Redis连接池
- **缓存策略**: 多层缓存架构
- **异步处理**: 消息队列异步处理
- **批处理**: 批量处理提高效率

### 监控优化
- **指标采集**: 合理的采集频率
- **数据保留**: 根据需求设置保留期
- **告警优化**: 避免告警风暴
- **仪表盘**: 关键指标可视化

## 升级和维护

### 滚动升级
```bash
# 更新镜像版本
kubectl set image deployment/gateway gateway=voicehelper/gateway:v2.1.0 -n voicehelper

# 查看升级状态
kubectl rollout status deployment/gateway -n voicehelper

# 回滚到上一版本
kubectl rollout undo deployment/gateway -n voicehelper
```

### 配置更新
```bash
# 更新ConfigMap
kubectl patch configmap voicehelper-config -n voicehelper --patch '{"data":{"LOG_LEVEL":"debug"}}'

# 重启相关Pod
kubectl rollout restart deployment/gateway -n voicehelper
```

### 定期维护
- **证书更新**: SSL/TLS证书自动续期
- **镜像更新**: 定期更新基础镜像
- **安全补丁**: 及时应用安全更新
- **性能调优**: 根据监控数据优化配置

## 总结

本部署方案实现了VoiceHelper系统的完整Kubernetes部署，具有以下特点：

1. **完整性**: 不简化任何功能，保持系统完整性
2. **高可用**: 多副本、故障转移、自动恢复
3. **可扩展**: 自动扩缩容、水平扩展
4. **可观测**: 完整的监控、日志、追踪体系
5. **安全性**: 全面的安全配置和访问控制
6. **易维护**: 自动化部署、测试、升级流程

通过本指南，您可以在Kubernetes环境中成功部署和运维VoiceHelper系统，实现生产级的高可用部署。
