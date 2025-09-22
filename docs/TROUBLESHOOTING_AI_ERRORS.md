# VoiceHelper AI - 错误诊断和解决方案

## 🚨 问题描述

**用户报告**: 文本输入"你好，你是谁"，AI回复"抱歉，发生了错误，请稍后重试。"

## 🔍 问题诊断过程

### 1. 直接测试算法服务 ✅

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好，你是谁"}]}'
```

**结果**: 正常返回AI回复，算法服务工作正常。

### 2. 测试网关服务 ❌

```bash
curl -X POST http://localhost:8080/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好，你是谁", "user_id": "test_user"}'
```

**结果**: 404 Not Found - 网关缺少聊天路由配置。

### 3. 检查前端配置 ❌

**发现问题**: 
- 前端配置 `NEXT_PUBLIC_API_URL=http://localhost:8080`
- 在Docker容器内，`localhost` 指向容器自身，无法访问网关服务
- 导致前端无法连接到后端API

## 🛠️ 解决方案

### 方案1: 修复前端API配置 (已实施)

**问题**: 前端容器内无法通过 localhost 访问网关服务

**解决**: 修改环境变量使用Docker服务名
```bash
# 修改 env.example
NEXT_PUBLIC_API_URL=http://gateway:8080
NEXT_PUBLIC_WS_URL=ws://gateway:8080
NEXT_PUBLIC_VOICE_WS_URL=ws://voice-service:8001
```

### 方案2: 网关添加聊天路由 (待实施)

**问题**: 网关服务缺少 `/api/v1/chat` 路由

**解决**: 需要在网关服务中添加代理到算法服务的路由配置

### 方案3: 直接使用算法服务 (临时方案)

**临时解决**: 前端直接连接算法服务
```bash
NEXT_PUBLIC_API_URL=http://algo-service:8000
```

## 📋 完整解决步骤

### 步骤1: 修复前端API配置

```bash
# 1. 修改环境变量
vim env.example

# 2. 重启前端服务
docker-compose -f docker-compose.local.yml up -d frontend
```

### 步骤2: 验证服务连通性

```bash
# 测试容器间网络连通性
docker exec voicehelper-frontend curl http://gateway:8080/health
docker exec voicehelper-frontend curl http://algo-service:8000/health
```

### 步骤3: 测试AI功能

```bash
# 直接测试算法服务
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好，你是谁"}]}'
```

## 🔧 网关路由配置 (推荐实施)

为了完整解决问题，建议在网关服务中添加以下路由：

### Go 网关路由示例

```go
// 在网关服务中添加聊天路由
func setupChatRoutes(r *gin.Engine) {
    api := r.Group("/api/v1")
    {
        // 聊天接口 - 代理到算法服务
        api.POST("/chat", func(c *gin.Context) {
            var req struct {
                Message string `json:"message"`
                UserID  string `json:"user_id"`
            }
            
            if err := c.ShouldBindJSON(&req); err != nil {
                c.JSON(400, gin.H{"error": "Invalid request"})
                return
            }
            
            // 转换为算法服务格式
            algoReq := map[string]interface{}{
                "messages": []map[string]string{
                    {"role": "user", "content": req.Message},
                },
            }
            
            // 代理到算法服务
            proxyToAlgoService(c, algoReq)
        })
        
        // 查询接口 - 直接代理
        api.POST("/query", func(c *gin.Context) {
            var req map[string]interface{}
            if err := c.ShouldBindJSON(&req); err != nil {
                c.JSON(400, gin.H{"error": "Invalid request"})
                return
            }
            proxyToAlgoService(c, req)
        })
    }
}

func proxyToAlgoService(c *gin.Context, data interface{}) {
    // 发送请求到算法服务
    resp, err := http.Post("http://algo-service:8000/query", 
        "application/json", 
        bytes.NewBuffer(jsonData))
    
    if err != nil {
        c.JSON(500, gin.H{"error": "Service unavailable"})
        return
    }
    
    // 流式返回响应
    c.Stream(func(w io.Writer) bool {
        // 处理流式响应
        return true
    })
}
```

## 🌐 Nginx 反向代理配置

另一个解决方案是通过 Nginx 直接代理到算法服务：

```nginx
# 在 nginx.conf 中添加
location /api/v1/chat {
    proxy_pass http://algo-service:8000/query;
    proxy_set_header Content-Type application/json;
    proxy_buffering off;
    proxy_cache off;
}

location /api/v1/query {
    proxy_pass http://algo-service:8000/query;
    proxy_set_header Content-Type application/json;
    proxy_buffering off;
    proxy_cache off;
}
```

## 🧪 测试验证

### 测试脚本

```bash
#!/bin/bash
echo "=== VoiceHelper AI 错误诊断测试 ==="

# 1. 测试算法服务
echo "1. 测试算法服务..."
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "你好，你是谁"}]}' | head -5

# 2. 测试网关健康检查
echo -e "\n2. 测试网关服务..."
curl -s http://localhost:8080/health | jq .

# 3. 测试前端访问
echo -e "\n3. 测试前端服务..."
curl -s -I http://localhost:3000 | head -1

# 4. 测试容器间连通性
echo -e "\n4. 测试容器间连通性..."
docker exec voicehelper-frontend curl -s http://gateway:8080/health | jq .status

echo -e "\n=== 测试完成 ==="
```

## 📊 错误监控

### 日志监控命令

```bash
# 实时监控所有服务日志
docker-compose -f docker-compose.local.yml logs -f

# 监控特定服务
docker logs voicehelper-frontend -f
docker logs voicehelper-gateway -f  
docker logs voicehelper-algo -f

# 查看错误日志
docker logs voicehelper-frontend 2>&1 | grep -i error
```

### 常见错误模式

| 错误信息 | 可能原因 | 解决方案 |
|----------|----------|----------|
| `ECONNREFUSED` | 服务连接被拒绝 | 检查服务名和端口配置 |
| `404 Not Found` | 路由不存在 | 检查API路径和网关配置 |
| `500 Internal Server Error` | 服务内部错误 | 查看服务日志定位问题 |
| `抱歉，发生了错误` | 前端错误处理 | 检查前端API配置和网络连通性 |

## 🚀 预防措施

### 1. 健康检查监控

```yaml
# docker-compose.yml 中添加健康检查
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
```

### 2. 错误处理改进

```javascript
// 前端错误处理改进
const apiCall = async (endpoint, data) => {
  try {
    const response = await fetch(`${API_URL}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return response;
  } catch (error) {
    console.error('API调用失败:', error);
    // 提供更具体的错误信息
    throw new Error(`连接失败: ${error.message}`);
  }
};
```

### 3. 配置验证

```bash
# 启动时验证配置
echo "验证服务配置..."
echo "前端API地址: $NEXT_PUBLIC_API_URL"
echo "网关地址: http://gateway:8080"
echo "算法服务地址: http://algo-service:8000"
```

## 📝 总结

**根本原因**: 
1. 前端容器内使用 `localhost` 无法访问其他容器服务
2. 网关服务缺少聊天相关的API路由配置

**解决方案**:
1. ✅ 修改前端API配置使用Docker服务名
2. 🔄 添加网关路由配置 (推荐)
3. 🔄 改进错误处理和监控

**验证方法**:
- 直接测试算法服务接口
- 检查容器间网络连通性
- 监控服务日志和错误信息

---

*最后更新时间: 2025-09-22*  
*问题状态: 部分解决，需要完善网关路由配置*
