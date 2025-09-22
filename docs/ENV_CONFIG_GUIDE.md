# 环境变量配置指南

## 概述

本指南说明如何正确配置 VoiceHelper AI 系统的环境变量，特别是算法服务所需的 API 密钥。

## 当前状态

✅ **环境变量加载机制已配置完成**
- Docker Compose 已配置 `env_file: .env`
- 环境变量正确传递到算法服务容器
- 容器中已加载 64 个环境变量

## 配置步骤

### 1. 环境变量文件位置

环境变量文件位于项目根目录：
```
/Users/lintao/important/ai-customer/voicehelper/.env
```

### 2. 当前配置状态

| 环境变量 | 当前值 | 状态 | 说明 |
|---------|--------|------|------|
| `ARK_API_KEY` | `1a208824-2b22-4a7f-ac89-49c4b1dcc5a7` | ❌ 需要更新 | 当前密钥格式不正确或无权限 |
| `ARK_BASE_URL` | `https://ark.cn-beijing.volces.com/api/v3` | ✅ 正确 | 豆包 API 基础 URL |
| `ARK_MODEL` | `ep-20241201140014-vbzjz` | ❌ 需要验证 | 模型不存在或无访问权限 |
| `OPENAI_API_KEY` | `your-openai-api-key-here` | ❌ 占位符 | 需要真实的 OpenAI API 密钥 |

### 3. 如何获取正确的 ARK API 密钥

#### 3.1 访问火山引擎控制台
1. 登录 [火山引擎控制台](https://console.volcengine.com/)
2. 进入 "豆包大模型" 或 "ARK" 服务
3. 在 API 管理页面获取 API 密钥

#### 3.2 ARK API 密钥格式
正确的 ARK API 密钥通常格式为：
- 长度：通常 32-64 字符
- 格式：可能以特定前缀开头（如 `ak-` 或其他）
- 示例：`ak-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

#### 3.3 模型 ID 配置
1. 在控制台查看可用的模型列表
2. 确认模型 ID 和访问权限
3. 常见的豆包模型 ID 格式：`ep-xxxxxxxxxxxxxxxxx`

### 4. 配置更新步骤

#### 4.1 更新 .env 文件
```bash
# 编辑环境变量文件
vim .env

# 更新以下配置
ARK_API_KEY=your-real-ark-api-key-here
ARK_MODEL=your-correct-model-id
OPENAI_API_KEY=your-openai-api-key-or-leave-empty
```

#### 4.2 重启服务
```bash
# 重启算法服务以加载新的环境变量
docker-compose -f docker-compose.local.yml restart algo-service
```

#### 4.3 验证配置
```bash
# 检查环境变量是否正确加载
docker-compose -f docker-compose.local.yml exec algo-service env | grep ARK_

# 测试 API 功能
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好"}],
    "dataset_id": "test",
    "top_k": 3
  }'
```

## 故障排除

### 问题 1：404 错误 "InvalidEndpointOrModel.NotFound"
**原因**：模型 ID 不存在或没有访问权限
**解决方案**：
1. 检查控制台中的可用模型列表
2. 确认账户有权限访问该模型
3. 更新 `ARK_MODEL` 为正确的模型 ID

### 问题 2：401 错误 "AuthenticationError"
**原因**：API 密钥格式不正确或已过期
**解决方案**：
1. 重新生成 API 密钥
2. 检查密钥格式是否正确
3. 确认密钥未过期

### 问题 3：环境变量未加载
**原因**：Docker Compose 配置问题
**解决方案**：
1. 确认 `.env` 文件存在于项目根目录
2. 检查 `docker-compose.local.yml` 中的 `env_file` 配置
3. 重启容器服务

## 当前系统状态

### ✅ 已完成
- [x] Docker Compose 环境变量加载配置
- [x] 环境变量正确传递到容器
- [x] 算法服务正常启动和运行
- [x] 本地向量存储功能正常
- [x] 文档入库和检索功能正常

### ⚠️ 需要配置
- [ ] 正确的 ARK API 密钥
- [ ] 有效的模型 ID
- [ ] OpenAI API 密钥（可选）

### 🔧 验证命令
```bash
# 检查服务健康状态
curl -s "http://localhost:8000/health" | jq .

# 检查环境变量
docker-compose -f docker-compose.local.yml exec algo-service env | grep -E "ARK_|OPENAI_"

# 测试 API 功能
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "测试"}], "dataset_id": "test"}'
```

## 安全注意事项

1. **不要提交真实的 API 密钥到代码仓库**
2. **定期轮换 API 密钥**
3. **限制 API 密钥的访问权限**
4. **在生产环境中使用更安全的密钥管理方案**

---

*最后更新：2025-09-22*
