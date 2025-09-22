# 豆包大模型配置指南

## 当前状态分析

### 测试结果
经过测试，当前的 ARK API 配置存在以下问题：

```
API Key: 1a208824-2b22-4a7f-ac89-49c4b1dcc5a7
Base URL: https://ark.cn-beijing.volces.com/api/v3
Current Model: ep-20241201140014-vbzjz
```

**所有测试的模型都返回 404 错误**：
- ❌ doubao-pro-4k
- ❌ doubao-lite-4k  
- ❌ doubao-pro-32k
- ❌ doubao-lite-32k
- ❌ doubao-pro-128k
- ❌ doubao-lite-128k
- ❌ ep-20240611125343-9vnlk
- ❌ ep-20241201140014-vbzjz (当前模型)

## 问题分析

### 可能的原因
1. **API 密钥格式不正确**：当前密钥可能不是有效的火山引擎 ARK API 密钥
2. **账户权限问题**：账户可能没有访问豆包大模型的权限
3. **API 端点错误**：可能需要使用不同的 API 端点
4. **模型 ID 过期**：模型 ID 可能已经更新或不再可用

## 解决方案

### 1. 获取正确的 ARK API 密钥

#### 步骤 1：访问火山引擎控制台
1. 访问 [火山引擎控制台](https://console.volcengine.com/)
2. 登录您的账户
3. 导航到 "豆包大模型" 或 "ARK" 服务

#### 步骤 2：申请模型访问权限
1. 在控制台中申请豆包大模型的访问权限
2. 等待审核通过（可能需要几个工作日）
3. 确认账户已开通相关服务

#### 步骤 3：获取 API 密钥
1. 在 API 管理页面创建新的 API 密钥
2. 复制完整的 API 密钥（通常比当前的更长）
3. 记录可用的模型 ID 列表

### 2. 常见的豆包大模型 ID

根据火山引擎文档，常见的模型 ID 包括：

```bash
# 豆包 Pro 系列
doubao-pro-4k          # 4K 上下文长度
doubao-pro-32k         # 32K 上下文长度  
doubao-pro-128k        # 128K 上下文长度

# 豆包 Lite 系列
doubao-lite-4k         # 轻量版 4K 上下文
doubao-lite-32k        # 轻量版 32K 上下文
doubao-lite-128k       # 轻量版 128K 上下文

# 端点模型（需要特定权限）
ep-xxxxxxxxxxxxxxxxx   # 自定义端点模型
```

### 3. API 配置更新

#### 更新 .env 文件
```bash
# 豆包 API 配置
ARK_API_KEY=your-new-valid-ark-api-key-here
ARK_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
ARK_MODEL=doubao-pro-4k  # 推荐使用的模型

# 或者根据需求选择其他模型
# ARK_MODEL=doubao-lite-4k     # 轻量版，成本更低
# ARK_MODEL=doubao-pro-32k     # 更长的上下文
# ARK_MODEL=doubao-pro-128k    # 最长的上下文
```

#### 重启服务
```bash
# 重启算法服务以加载新配置
docker-compose -f docker-compose.local.yml restart algo-service
```

### 4. 验证配置

#### 测试脚本
```bash
# 在容器中测试新的 API 配置
docker-compose -f docker-compose.local.yml exec algo-service python3 -c "
import os
import requests

ark_api_key = os.getenv('ARK_API_KEY')
ark_base_url = os.getenv('ARK_BASE_URL') 
ark_model = os.getenv('ARK_MODEL')

headers = {
    'Authorization': f'Bearer {ark_api_key}',
    'Content-Type': 'application/json'
}

payload = {
    'model': ark_model,
    'messages': [{'role': 'user', 'content': '你好，请简单介绍一下自己'}],
    'temperature': 0.7,
    'max_tokens': 100
}

response = requests.post(f'{ark_base_url}/chat/completions', 
                        headers=headers, json=payload, timeout=10)

print(f'Status: {response.status_code}')
if response.status_code == 200:
    result = response.json()
    print(f'Response: {result[\"choices\"][0][\"message\"][\"content\"]}')
else:
    print(f'Error: {response.text}')
"
```

#### API 测试
```bash
# 测试查询功能
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user", 
        "content": "什么是人工智能？"
      }
    ],
    "dataset_id": "test",
    "top_k": 3,
    "temperature": 0.7,
    "max_tokens": 200
  }'
```

## 推荐配置

### 生产环境推荐
```bash
# 高性能配置
ARK_MODEL=doubao-pro-4k      # 平衡性能和成本
ARK_MODEL=doubao-pro-32k     # 需要更长上下文时使用
```

### 开发/测试环境推荐  
```bash
# 成本优化配置
ARK_MODEL=doubao-lite-4k     # 降低开发成本
```

## 故障排除

### 常见错误及解决方案

#### 1. 404 错误 "InvalidEndpointOrModel.NotFound"
- **原因**：模型 ID 不存在或无权限访问
- **解决**：检查控制台中的可用模型列表，使用正确的模型 ID

#### 2. 401 错误 "AuthenticationError"  
- **原因**：API 密钥无效或格式错误
- **解决**：重新生成 API 密钥，确保格式正确

#### 3. 403 错误 "PermissionDenied"
- **原因**：账户没有访问权限
- **解决**：在控制台申请相应的模型访问权限

#### 4. 429 错误 "RateLimitExceeded"
- **原因**：请求频率超限
- **解决**：降低请求频率或升级配额

## 下一步行动

1. **立即行动**：
   - [ ] 访问火山引擎控制台
   - [ ] 申请豆包大模型访问权限
   - [ ] 获取有效的 API 密钥

2. **配置更新**：
   - [ ] 更新 `.env` 文件中的 API 密钥
   - [ ] 选择合适的模型 ID
   - [ ] 重启服务并测试

3. **验证测试**：
   - [ ] 运行 API 测试脚本
   - [ ] 测试查询功能
   - [ ] 验证模型响应质量

## 联系支持

如果遇到问题，可以：
- 查看火山引擎官方文档
- 联系火山引擎技术支持
- 在开发者社区寻求帮助

---

*最后更新：2025-09-22*
*状态：需要获取有效的 ARK API 密钥*
