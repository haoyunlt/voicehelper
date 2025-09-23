# 🚀 国内大模型快速配置指南

## 一键配置 (推荐)

### 步骤 1: 运行配置脚本
```bash
./setup_multi_model.sh
```

选择 **方案 2: 性能平衡方案** (推荐)

### 步骤 2: 获取 API 密钥

#### 豆包大模型 (主要模型)
1. 访问: https://console.volcengine.com/
2. 注册/登录 → 开通豆包大模型服务
3. 创建 API 密钥
4. 复制密钥到 `.env` 文件的 `ARK_API_KEY`

#### GLM-4 (备用模型)
1. 访问: https://open.bigmodel.cn/
2. 注册 → 创建 API Key
3. 复制密钥到 `.env` 文件的 `GLM_API_KEY`

### 步骤 3: 启动服务
```bash
# 重新构建和启动
docker-compose -f docker-compose.local.yml build algo-service
docker-compose -f docker-compose.local.yml up -d

# 测试服务
curl http://localhost:8000/health
```

## 手动配置

如果需要手动配置，编辑 `.env` 文件：

```bash
# 性能平衡方案
PRIMARY_MODEL=doubao-pro-4k

# 豆包大模型 (主要)
ARK_API_KEY=your-real-ark-api-key-here
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL=doubao-pro-4k

# GLM-4 (备用)
GLM_API_KEY=your-real-glm-api-key-here
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4

# 其他配置保持不变...
```

## 验证配置

```bash
# 测试模型调用
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好，请简单介绍一下自己"}],
    "dataset_id": "test",
    "max_tokens": 100
  }'

# 查看模型统计
curl http://localhost:8000/models/stats
```

## 成本对比

| 配置方案 | 月成本 (100万tokens) | 适用场景 |
|----------|---------------------|----------|
| 成本优化 | ~1.1元 | 个人项目 |
| 性能平衡 | ~3.0元 | 生产环境 |
| 企业方案 | ~5.0元 | 高可用需求 |

**对比 OpenAI GPT-4: 节省 90%+ 成本！**

## 问题排查

### 常见问题
1. **API 密钥无效**: 检查密钥格式和权限
2. **模型不存在**: 确认账户已开通相应服务
3. **网络连接**: 检查防火墙和网络设置

### 获取帮助
- 查看详细文档: `docs/DOMESTIC_LLM_RESEARCH.md`
- 查看配置指南: `docs/MODEL_SELECTION_SUMMARY.md`
- 运行诊断: `docs/ARK_API_DIAGNOSIS.md`

---

**🎯 推荐配置: 豆包 Pro + GLM-4 Flash**  
**💰 预期节省: 90%+ API 成本**  
**⚡ 集成难度: 零修改**
