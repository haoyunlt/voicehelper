# VoiceHelper Kubernetes 部署指南

## 概述

本目录包含 VoiceHelper 在 Kubernetes 上的完整部署配置，使用 BGE+FAISS 替代 Milvus 作为向量存储解决方案。

## 架构变更

### 向量存储架构
- **之前**: Milvus + etcd + MinIO
- **现在**: BGE (向量化) + FAISS (搜索)

### 优势
- 更轻量级的部署
- 减少外部依赖
- 更好的性能和资源利用率
- 简化的运维管理

## 部署文件说明

### 核心部署文件
- `00-namespace.yaml` - 命名空间和基础配置
- `01-configmap-secrets.yaml` - 配置映射和密钥
- `02-third-party-services.yaml` - 第三方服务（PostgreSQL, Redis, MinIO, NATS）
- `03-vector-services-bge-faiss.yaml` - BGE+FAISS 向量服务
- `04-application-services.yaml` - 应用服务
- `05-monitoring-services.yaml` - 监控服务
- `06-ingress-loadbalancer.yaml` - Ingress 和负载均衡
- `07-persistent-volumes.yaml` - 持久化卷

### 部署脚本
- `deploy.sh` - 主部署脚本
- `test-bge-faiss.sh` - BGE+FAISS 服务测试脚本
- `test-services.sh` - 服务测试脚本

## 快速开始

### 1. 环境准备

确保已安装以下工具：
```bash
# Kubernetes 集群
kubectl cluster-info

# Docker（用于构建镜像）
docker --version

# 其他工具
curl --version
jq --version
```

### 2. 构建镜像

```bash
# 构建 BGE 服务镜像
cd algo
docker build -f Dockerfile.bge -t voicehelper/bge-service:latest .

# 构建 FAISS 服务镜像
docker build -f Dockerfile.faiss -t voicehelper/faiss-service:latest .

# 构建其他服务镜像
cd ../backend
docker build -t voicehelper/gateway:latest .

cd ../platforms/web
docker build -t voicehelper/frontend:latest .
```

### 3. 部署服务

```bash
# 完整部署
./tools/deployment/k8s/deploy.sh full

# 最小部署（不包含监控）
./tools/deployment/k8s/deploy.sh minimal

# 仅部署监控
./tools/deployment/k8s/deploy.sh monitoring-only
```

### 4. 验证部署

```bash
# 检查所有服务状态
kubectl get pods -n voicehelper

# 测试 BGE+FAISS 服务
./tools/deployment/k8s/test-bge-faiss.sh

# 测试所有服务
./tools/deployment/k8s/test-services.sh
```

## 服务配置

### BGE 向量化服务
- **端口**: 8080
- **模型**: BAAI/bge-large-zh-v1.5
- **功能**: 文本向量化
- **API**: `/embed`, `/health`, `/ready`, `/metrics`

### FAISS 向量搜索服务
- **端口**: 8081
- **索引类型**: IVF (可配置)
- **功能**: 向量相似度搜索
- **API**: `/add`, `/search`, `/stats`, `/health`

## 访问地址

部署完成后，可通过以下地址访问：

### 应用服务
- 主应用: http://voicehelper.local
- API服务: http://api.voicehelper.local
- WebSocket: ws://ws.voicehelper.local
- 语音服务: http://voice.voicehelper.local

### 向量服务
- BGE服务: http://bge.voicehelper.local
- FAISS服务: http://faiss.voicehelper.local

### 监控服务
- Grafana: http://grafana.voicehelper.local (admin/VoiceHelper2025!)
- Prometheus: http://prometheus.voicehelper.local
- Jaeger: http://jaeger.voicehelper.local

### 管理工具
- MinIO Console: http://minio.voicehelper.local

## 配置 hosts 文件

将以下内容添加到 `/etc/hosts` 文件：

```
127.0.0.1 voicehelper.local
127.0.0.1 api.voicehelper.local
127.0.0.1 ws.voicehelper.local
127.0.0.1 voice.voicehelper.local
127.0.0.1 bge.voicehelper.local
127.0.0.1 faiss.voicehelper.local
127.0.0.1 grafana.voicehelper.local
127.0.0.1 prometheus.voicehelper.local
127.0.0.1 jaeger.voicehelper.local
127.0.0.1 minio.voicehelper.local
```

## 扩缩容配置

### 自动扩缩容
- BGE 服务: 1-4 副本，基于 CPU/内存使用率
- Gateway: 2-10 副本，基于 CPU/内存使用率
- Algo 服务: 1-6 副本，基于 CPU/内存使用率

### 手动扩缩容
```bash
# 扩展 BGE 服务
kubectl scale deployment bge-service -n voicehelper --replicas=3

# 扩展 FAISS 服务（StatefulSet）
kubectl scale statefulset faiss-service -n voicehelper --replicas=2
```

## 监控和日志

### Prometheus 指标
- BGE 服务: `/metrics` 端点
- FAISS 服务: `/metrics` 端点
- 自定义指标: 请求数量、处理时间、向量数量等

### 日志查看
```bash
# 查看 BGE 服务日志
kubectl logs -n voicehelper -l app.kubernetes.io/name=bge-service -f

# 查看 FAISS 服务日志
kubectl logs -n voicehelper -l app.kubernetes.io/name=faiss-service -f
```

## 故障排除

### 常见问题

1. **BGE 模型下载失败**
   ```bash
   # 检查网络连接和存储空间
   kubectl describe pod -n voicehelper -l app.kubernetes.io/name=bge-service
   ```

2. **FAISS 索引初始化失败**
   ```bash
   # 检查持久化卷和权限
   kubectl get pvc -n voicehelper
   ```

3. **服务无法访问**
   ```bash
   # 检查 Ingress 和服务状态
   kubectl get ingress -n voicehelper
   kubectl get svc -n voicehelper
   ```

### 调试命令
```bash
# 进入 BGE 服务容器
kubectl exec -it -n voicehelper deployment/bge-service -- /bin/bash

# 进入 FAISS 服务容器
kubectl exec -it -n voicehelper statefulset/faiss-service -- /bin/bash

# 查看事件
kubectl get events -n voicehelper --sort-by='.lastTimestamp'
```

## 备份和恢复

### FAISS 索引备份
```bash
# 手动触发保存
curl -X POST http://faiss.voicehelper.local/save

# 备份持久化卷数据
kubectl exec -n voicehelper statefulset/faiss-service -- tar -czf /tmp/faiss-backup.tar.gz /app/data
```

### 配置备份
```bash
# 导出配置
kubectl get configmap -n voicehelper -o yaml > voicehelper-config-backup.yaml
kubectl get secret -n voicehelper -o yaml > voicehelper-secrets-backup.yaml
```

## 清理部署

```bash
# 清理所有资源
./tools/deployment/k8s/deploy.sh cleanup

# 手动清理
kubectl delete namespace voicehelper
kubectl delete namespace voicehelper-monitoring
```

## 性能调优

### BGE 服务优化
- 调整 `MAX_BATCH_SIZE` 环境变量
- 配置 GPU 支持（如果可用）
- 调整副本数量

### FAISS 服务优化
- 选择合适的索引类型（IVF, HNSW, Flat）
- 调整 `FAISS_NLIST` 和 `FAISS_NPROBE` 参数
- 配置定期保存间隔

## 更新和升级

### 滚动更新
```bash
# 更新 BGE 服务镜像
kubectl set image deployment/bge-service -n voicehelper bge-service=voicehelper/bge-service:v2.0

# 更新 FAISS 服务镜像
kubectl set image statefulset/faiss-service -n voicehelper faiss-service=voicehelper/faiss-service:v2.0
```

### 配置更新
```bash
# 更新配置映射
kubectl apply -f 01-configmap-secrets.yaml

# 重启相关服务
kubectl rollout restart deployment/bge-service -n voicehelper
```

## 支持和贡献

如有问题或建议，请提交 Issue 或 Pull Request。

## 许可证

本项目采用 MIT 许可证。