# VoiceHelper Kubernetes 完整架构重设计

## 设计目标

1. **完整性** - 覆盖所有服务模块，不简化功能
2. **本地化** - 所有第三方软件本地部署，减少外部依赖
3. **可观测性** - 完整的监控、日志、链路追踪系统
4. **高可用** - 多副本、故障转移、自动恢复
5. **安全性** - 网络隔离、RBAC、密钥管理
6. **可扩展** - 自动扩缩容、资源优化

## 完整服务架构

### 1. 核心应用层
```
┌─────────────────────────────────────────────────────────────┐
│                     应用服务层                                │
├─────────────────────────────────────────────────────────────┤
│ Frontend (Web)    │ Gateway (Go)     │ Admin Portal        │
│ - Next.js         │ - API Gateway    │ - Management UI     │
│ - 3 replicas      │ - 3 replicas     │ - 2 replicas        │
├─────────────────────────────────────────────────────────────┤
│ Algo Service      │ Voice Service    │ Developer Portal    │
│ - Python FastAPI  │ - WebRTC/WebSocket│ - API Documentation │
│ - 3 replicas      │ - 2 replicas     │ - 1 replica         │
└─────────────────────────────────────────────────────────────┘
```

### 2. AI/ML 服务层
```
┌─────────────────────────────────────────────────────────────┐
│                    AI/ML 服务层                              │
├─────────────────────────────────────────────────────────────┤
│ BGE Service       │ FAISS Service    │ LLM Proxy Service   │
│ - 向量化服务       │ - 向量搜索       │ - 多模型路由         │
│ - 2 replicas      │ - StatefulSet    │ - 2 replicas        │
├─────────────────────────────────────────────────────────────┤
│ Speech Service    │ TTS Service      │ Reasoning Service   │
│ - ASR 语音识别     │ - 语音合成       │ - 推理引擎          │
│ - 2 replicas      │ - 2 replicas     │ - 2 replicas        │
└─────────────────────────────────────────────────────────────┘
```

### 3. 数据存储层
```
┌─────────────────────────────────────────────────────────────┐
│                    数据存储层                                │
├─────────────────────────────────────────────────────────────┤
│ PostgreSQL        │ Redis Cluster    │ MinIO Object Store  │
│ - 主从复制         │ - 3 节点集群     │ - 分布式存储         │
│ - 1 master + 2 read│ - 高可用缓存     │ - 4 节点集群         │
├─────────────────────────────────────────────────────────────┤
│ Neo4j Graph DB    │ ClickHouse       │ Elasticsearch       │
│ - 图数据库         │ - 时序数据       │ - 日志搜索          │
│ - 3 节点集群       │ - 分析数据库     │ - 3 节点集群         │
└─────────────────────────────────────────────────────────────┘
```

### 4. 消息队列层
```
┌─────────────────────────────────────────────────────────────┐
│                   消息队列层                                 │
├─────────────────────────────────────────────────────────────┤
│ NATS JetStream    │ Apache Kafka     │ RabbitMQ            │
│ - 实时消息         │ - 事件流处理     │ - 任务队列          │
│ - 3 节点集群       │ - 3 节点集群     │ - 3 节点集群         │
└─────────────────────────────────────────────────────────────┘
```

### 5. 监控可观测性层
```
┌─────────────────────────────────────────────────────────────┐
│                 监控可观测性层                               │
├─────────────────────────────────────────────────────────────┤
│ Prometheus        │ Grafana          │ AlertManager        │
│ - 指标收集         │ - 可视化仪表盘   │ - 告警管理          │
│ - 2 replicas      │ - 2 replicas     │ - 3 replicas        │
├─────────────────────────────────────────────────────────────┤
│ Jaeger           │ Fluentd          │ OpenTelemetry       │
│ - 链路追踪        │ - 日志收集       │ - 可观测性代理      │
│ - All-in-one     │ - DaemonSet      │ - DaemonSet         │
└─────────────────────────────────────────────────────────────┘
```

### 6. 服务网格层
```
┌─────────────────────────────────────────────────────────────┐
│                   服务网格层                                 │
├─────────────────────────────────────────────────────────────┤
│ Istio Service Mesh│ Envoy Proxy      │ Cert-Manager        │
│ - 流量管理         │ - 边车代理       │ - 证书管理          │
│ - 安全策略         │ - 负载均衡       │ - 自动续期          │
└─────────────────────────────────────────────────────────────┘
```

## 部署文件结构

```
tools/deployment/k8s/
├── 00-prerequisites/           # 前置条件
│   ├── namespace.yaml
│   ├── rbac.yaml
│   ├── network-policies.yaml
│   └── storage-classes.yaml
├── 01-infrastructure/          # 基础设施
│   ├── configmaps.yaml
│   ├── secrets.yaml
│   ├── persistent-volumes.yaml
│   └── service-accounts.yaml
├── 02-storage-services/        # 存储服务
│   ├── postgresql-cluster.yaml
│   ├── redis-cluster.yaml
│   ├── minio-cluster.yaml
│   ├── neo4j-cluster.yaml
│   ├── clickhouse-cluster.yaml
│   └── elasticsearch-cluster.yaml
├── 03-messaging-services/      # 消息队列
│   ├── nats-jetstream.yaml
│   ├── kafka-cluster.yaml
│   └── rabbitmq-cluster.yaml
├── 04-ai-ml-services/         # AI/ML服务
│   ├── bge-service.yaml
│   ├── faiss-service.yaml
│   ├── llm-proxy.yaml
│   ├── speech-service.yaml
│   ├── tts-service.yaml
│   └── reasoning-service.yaml
├── 05-application-services/    # 应用服务
│   ├── gateway.yaml
│   ├── algo-service.yaml
│   ├── voice-service.yaml
│   ├── frontend.yaml
│   ├── admin-portal.yaml
│   └── developer-portal.yaml
├── 06-monitoring-stack/        # 监控栈
│   ├── prometheus.yaml
│   ├── grafana.yaml
│   ├── alertmanager.yaml
│   ├── jaeger.yaml
│   ├── fluentd.yaml
│   └── opentelemetry.yaml
├── 07-service-mesh/           # 服务网格
│   ├── istio-base.yaml
│   ├── istio-control-plane.yaml
│   ├── istio-gateways.yaml
│   └── virtual-services.yaml
├── 08-ingress-networking/     # 网络入口
│   ├── nginx-ingress.yaml
│   ├── cert-manager.yaml
│   ├── ingress-rules.yaml
│   └── load-balancers.yaml
├── 09-security/               # 安全配置
│   ├── pod-security-policies.yaml
│   ├── network-policies.yaml
│   ├── rbac-policies.yaml
│   └── security-contexts.yaml
└── 10-operators/              # 操作器
    ├── postgres-operator.yaml
    ├── redis-operator.yaml
    ├── kafka-operator.yaml
    └── monitoring-operator.yaml
```

## 资源规划

### CPU/内存需求
- **总CPU需求**: ~50 cores
- **总内存需求**: ~128GB
- **存储需求**: ~2TB SSD

### 节点规划
- **Master节点**: 3个 (4C8G)
- **Worker节点**: 6个 (16C32G)
- **存储节点**: 3个 (8C16G + 1TB SSD)

## 网络架构

### 服务发现
- 内部DNS: CoreDNS
- 服务注册: Kubernetes Service
- 负载均衡: Istio + Envoy

### 网络策略
- 命名空间隔离
- 微分段网络
- 入口/出口流量控制

### 安全策略
- mTLS 加密
- RBAC 权限控制
- Pod Security Standards

## 部署策略

### 1. 蓝绿部署
- 零停机更新
- 快速回滚
- 流量切换

### 2. 金丝雀发布
- 渐进式发布
- A/B 测试
- 风险控制

### 3. 滚动更新
- 默认更新策略
- 健康检查
- 自动回滚

## 监控指标

### 应用指标
- QPS/TPS
- 响应时间
- 错误率
- 业务指标

### 基础设施指标
- CPU/内存使用率
- 磁盘I/O
- 网络流量
- 存储容量

### 自定义指标
- AI模型推理时间
- 向量搜索延迟
- 语音处理质量
- 用户体验指标

## 告警策略

### 关键告警
- 服务不可用
- 响应时间超阈值
- 错误率过高
- 资源耗尽

### 预警告警
- 资源使用率高
- 磁盘空间不足
- 证书即将过期
- 性能下降

## 备份恢复

### 数据备份
- 数据库定期备份
- 配置文件备份
- 应用状态备份
- 跨区域复制

### 灾难恢复
- RTO: 15分钟
- RPO: 5分钟
- 自动故障转移
- 数据一致性保证
