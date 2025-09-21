# 版本管理

## 当前版本状态

| 版本 | 状态 | 分支 | 发布日期 | 主要特性 |
|------|------|------|----------|----------|
| **v1.0.0** | ✅ 已发布 | main | 2025-09-21 | MVP核心功能完成 |
| **v1.1.0** | 🚧 开发中 | feat/sprint1-production-ready | 预计 Week 2 | 生产就绪 |
| **v1.2.0** | 📋 计划中 | feat/sprint2-ux-optimization | 预计 Week 4 | 体验优化 |
| **v1.3.0** | 📋 计划中 | feat/sprint3-feature-expansion | 预计 Week 6 | 功能扩展 |
| **v1.4.0** | 📋 计划中 | feat/sprint4-intelligence-upgrade | 预计 Week 8 | 智能化升级 |

---

## v1.1.0 - Sprint 1: 生产就绪 🚧

### 已完成功能
- ✅ PostgreSQL数据模型设计
  - 完整的表结构（租户、用户、会话、消息、审计日志等）
  - 索引优化和分区策略
  - 触发器和函数
  
- ✅ Redis缓存层实现
  - 会话缓存管理
  - 语义缓存机制
  - 热点数据缓存
  - 分布式锁支持
  
- ✅ JWT认证和安全加固
  - JWT中间件完善（自动续期、黑名单）
  - RBAC权限系统
  - 租户隔离中间件
  - API Key认证支持
  
- ✅ Kubernetes部署配置
  - 完整的K8s资源定义
  - HPA自动伸缩
  - PodDisruptionBudget
  - NetworkPolicy安全策略

### 待完成功能
- [ ] 敏感信息脱敏处理
- [ ] 完整的Grafana监控面板
- [ ] ELK日志聚合配置

### 技术债务
- [ ] 单元测试覆盖率提升到80%
- [ ] API文档自动生成
- [ ] 数据库迁移脚本

---

## v1.0.0 - MVP Release ✅

### 核心功能
- ✅ **双通道对话**
  - 文本SSE流式对话
  - WebSocket语音实时通信
  - 同会话无缝切换

- ✅ **语音处理**
  - ASR实时转写
  - TTS流式合成
  - VAD端点检测
  - Barge-in语音打断（<150ms）

- ✅ **智能检索**
  - Milvus向量数据库集成
  - LangChain RAG实现
  - 召回率≥85%

- ✅ **Agent能力**
  - LangGraph状态机Agent
  - 意图识别和执行计划
  - MCP工具协议（文件系统/HTTP/数据库/GitHub）
  - 完整审计日志

- ✅ **平台支持**
  - 微信小程序登录
  - JWT令牌管理
  - 多租户隔离

- ✅ **监控体系**
  - Prometheus指标
  - 链路追踪
  - 性能监控

### 性能指标
- 文本首token: <800ms ✅
- 语音首音: <500ms ✅
- 端到端首响: <700ms ✅
- Barge-in响应: ≤150ms ✅
- RAG召回率: ≥85% ✅

---

## 版本发布流程

### 1. 开发阶段
```bash
# 创建功能分支
git checkout -b feat/sprint{n}-{feature}

# 开发和测试
make test
make lint

# 提交代码
git add .
git commit -m "feat: {description}"
```

### 2. 集成阶段
```bash
# 创建PR到main
# 通过Code Review
# 运行CI/CD pipeline
```

### 3. 发布阶段
```bash
# 合并到main
git checkout main
git merge feat/sprint{n}-{feature}

# 创建版本标签
git tag -a v{version} -m "Release v{version}: {description}"
git push origin v{version}

# 部署到生产
kubectl apply -f deploy/k8s/
```

### 4. 回滚流程
```bash
# Helm回滚
helm rollback chatbot {revision}

# Git回滚
git revert {commit}
git push origin main
```

---

## 分支保护规则

### main分支
- 禁止直接push
- 需要PR和Code Review
- 必须通过所有CI检查
- 至少1名Code Owner批准

### 功能分支
- 命名规范: `feat/sprint{n}-{feature}`
- 定期同步main分支
- 完成后删除

### 热修复分支
- 命名规范: `hotfix/{issue}`
- 快速修复生产问题
- 合并后立即发布

---

## 版本兼容性矩阵

| 组件 | v1.0.0 | v1.1.0 | v1.2.0 | v1.3.0 | v1.4.0 |
|------|--------|--------|--------|--------|--------|
| Go | 1.21 | 1.21 | 1.21 | 1.22 | 1.22 |
| Python | 3.11 | 3.11 | 3.11 | 3.11 | 3.12 |
| Node.js | 18 | 18 | 18 | 20 | 20 |
| PostgreSQL | - | 15 | 15 | 15 | 16 |
| Redis | - | 7 | 7 | 7 | 7 |
| Milvus | 2.3.4 | 2.3.4 | 2.3.5 | 2.4.0 | 2.4.0 |
| Kubernetes | 1.27 | 1.27 | 1.28 | 1.28 | 1.29 |

---

## 变更日志格式

### 添加 (Added)
- 新功能

### 修改 (Changed)
- 功能改进

### 修复 (Fixed)
- Bug修复

### 删除 (Deprecated)
- 废弃功能

### 安全 (Security)
- 安全更新

---

*最后更新: 2025-09-21*
*当前开发版本: v1.1.0*
