# VoiceHelper 项目清理与修复任务清单

> 基于2025年9月22日的代码重构和部署过程中的修改生成

## 📊 当前状态概览

### ✅ 已完成的模块编译
- **后端Go模块**: 编译成功，简化版服务运行正常
- **算法Python模块**: 基础依赖安装完成，核心功能可用
- **前端Web模块**: Next.js构建成功，页面正常显示
- **管理后台模块**: Python依赖安装完成

### 🚀 当前服务运行状态
| 服务 | 端口 | 状态 | 备注 |
|------|------|------|------|
| 后端服务 | 8080 | ✅ 运行中 | 简化版本，基础API可用 |
| 前端服务 | 3000 | ✅ 运行中 | Next.js开发服务器 |
| 算法服务 | 8000 | ⚠️ 需修复 | 需要PYTHONPATH配置 |
| 管理后台 | 5001 | ❌ 失败 | 数据库连接问题 |

---

## 🔧 紧急修复任务 (Priority: High)

### 1. 恢复Prometheus指标系统
**状态**: 待处理  
**问题**: 删除了重复的指标定义，导致监控功能缺失  
**任务**:
- [ ] 重新设计统一的指标收集架构
- [ ] 创建`pkg/metrics/unified_metrics.go`
- [ ] 避免重复注册问题
- [ ] 恢复SSE、WebSocket、WebRTC相关指标

**影响文件**:
```
backend/pkg/metrics/unified_metrics.go (已删除)
backend/pkg/middleware/metrics.go (已删除)
backend/internal/handlers/chat_sse.go (指标调用已注释)
```

### 2. 修复管理后台数据库连接
**状态**: 待处理  
**问题**: PostgreSQL连接失败 - "could not translate host name 'postgres'"  
**任务**:
- [ ] 配置本地SQLite数据库作为替代方案
- [ ] 或配置本地PostgreSQL实例
- [ ] 更新数据库连接字符串
- [ ] 测试管理后台功能

**错误信息**:
```
sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) 
could not translate host name "postgres" to address: 
nodename nor servname provided, or not known
```

### 3. 修复算法服务模块路径
**状态**: 待处理  
**问题**: `ModuleNotFoundError: No module named 'core'`  
**任务**:
- [ ] 在算法服务启动脚本中设置PYTHONPATH
- [ ] 创建`algo/start.sh`启动脚本
- [ ] 更新Docker配置中的PYTHONPATH
- [ ] 验证所有模块导入正常

**当前解决方案**:
```bash
PYTHONPATH=/Users/lintao/important/ai-customer/voicehelper/algo python app/main.py
```

---

## 🚀 功能恢复任务 (Priority: Medium)

### 4. 重新实现语音WebSocket处理器
**状态**: 待处理  
**问题**: 删除了关键的语音处理文件  
**任务**:
- [ ] 恢复`backend/internal/ssews/voice_ws.go`
- [ ] 恢复`backend/internal/ssews/webrtc_signaling.go`
- [ ] 重新实现语音流处理逻辑
- [ ] 添加WebRTC信令处理

**已删除文件**:
```
backend/internal/ssews/voice_ws.go
backend/internal/ssews/webrtc_signaling.go
```

### 5. 添加中间件指标监控
**状态**: 待处理  
**问题**: 删除了指标中间件  
**任务**:
- [ ] 重新实现`pkg/middleware/metrics.go`
- [ ] 添加请求计数、延迟统计
- [ ] 集成到Gin路由中
- [ ] 配置Prometheus端点

### 6. 创建简化管理后台
**状态**: 待处理  
**问题**: 当前版本依赖外部PostgreSQL  
**任务**:
- [ ] 创建SQLite版本的管理后台
- [ ] 简化数据模型
- [ ] 保留核心管理功能
- [ ] 添加基础数据可视化

---

## 📚 文档和配置任务 (Priority: Low)

### 7. 恢复项目文档
**状态**: 待处理  
**问题**: 删除了重要的项目结构文档  
**任务**:
- [ ] 重新创建`docs/NEW_PROJECT_STRUCTURE.md`
- [ ] 重新创建`docs/OPTIMIZED_PROJECT_STRUCTURE.md`
- [ ] 更新项目架构说明
- [ ] 添加部署指南

**已删除文件**:
```
docs/NEW_PROJECT_STRUCTURE.md
docs/OPTIMIZED_PROJECT_STRUCTURE.md
```

### 8. 添加服务健康检查
**状态**: 待处理  
**任务**:
- [ ] 为所有服务添加统一的`/health`端点
- [ ] 实现服务依赖检查
- [ ] 添加服务状态监控
- [ ] 集成到Docker Compose健康检查

---

## 🔄 已删除的关键文件清单

### 后端文件
```
backend/pkg/metrics/unified_metrics.go
backend/pkg/middleware/metrics.go
backend/internal/ssews/voice_ws.go
backend/internal/ssews/webrtc_signaling.go
backend/pkg/metrics/collector.go
backend/pkg/metrics/voice_metrics.go
```

### 文档文件
```
docs/NEW_PROJECT_STRUCTURE.md
docs/OPTIMIZED_PROJECT_STRUCTURE.md
```

### 修改的文件
```
backend/internal/ssews/chat_sse.go (指标调用已注释)
backend/internal/ssews/v2_voice.go (类型引用已修复)
backend/internal/ssews/v2_chat.go (类型引用已修复)
backend/cmd/gateway/main.go (移除指标初始化)
```

---

## 📋 执行计划

### 第一阶段：基础修复 (1-2天)
1. 修复算法服务模块路径问题
2. 创建简化版管理后台
3. 添加基础健康检查

### 第二阶段：功能恢复 (3-5天)
1. 重新实现Prometheus指标系统
2. 恢复语音WebSocket处理器
3. 添加中间件监控

### 第三阶段：完善优化 (1-2天)
1. 恢复项目文档
2. 完善服务监控
3. 优化部署配置

---

## 🚨 注意事项

1. **指标系统重构**: 避免重复注册，使用单例模式
2. **数据库配置**: 优先使用SQLite，减少外部依赖
3. **模块路径**: 统一使用相对导入或正确设置PYTHONPATH
4. **服务启动**: 确保所有服务都有独立的启动脚本
5. **错误处理**: 添加优雅的错误处理和恢复机制

---

## 📞 联系信息

- **创建时间**: 2025年9月22日
- **最后更新**: 2025年9月22日
- **负责人**: 开发团队
- **优先级**: 按照上述分类执行

---

*此文档将随着任务进展持续更新*
