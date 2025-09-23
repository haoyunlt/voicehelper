# 聊天取消功能实现文档

## 功能概述

本文档描述了VoiceHelper算法服务模块中新实现的聊天取消功能。该功能允许用户在流式聊天过程中随时取消正在进行的对话，提供更好的用户体验和资源管理。

## 核心组件

### 1. ChatSessionManager 会话管理器

负责管理所有活跃的聊天会话，提供以下功能：

- **会话创建**: 为每个聊天请求创建唯一会话
- **会话跟踪**: 跟踪会话状态和活动时间
- **会话取消**: 通过事件机制取消正在进行的会话
- **自动清理**: 定期清理过期和无效会话

```python
class ChatSessionManager:
    def create_session(self, session_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]
    def cancel_session(self, session_id: str) -> bool
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]
    def cleanup_expired_sessions(self)
```

### 2. 会话状态管理

每个会话包含以下状态信息：

```python
session_info = {
    "session_id": str,           # 会话唯一标识
    "created_at": datetime,      # 创建时间
    "last_activity": datetime,   # 最后活动时间
    "status": str,              # 状态: active/cancelled/completed
    "request_data": dict,       # 请求数据
    "cancel_event": asyncio.Event,  # 取消事件
    "generator": object,        # 生成器对象
    "response_chunks": list     # 响应片段
}
```

## API接口

### 1. 流式聊天接口 (增强版)

**端点**: `POST /api/v1/chat/stream`

**功能增强**:
- 自动创建会话管理
- 实时检查取消状态
- 优雅处理取消事件

```python
@app.post("/api/v1/chat/stream")
async def stream_chat(request: ChatRequest):
    # 创建会话
    session_info = session_manager.create_session(request.session_id, {...})
    
    # 流式处理，支持取消检查
    for result in agent_graph.stream(request.query, cb=callback):
        if session_info["cancel_event"].is_set():
            # 发送取消事件并退出
            yield {"event": "cancelled", "data": {...}}
            break
        # 正常处理...
```

### 2. 取消聊天接口

**端点**: `POST /api/v1/chat/cancel`

**请求格式**:
```json
{
    "session_id": "your_session_id"
}
```

**响应格式**:
```json
{
    "status": "cancelled",
    "session_id": "your_session_id",
    "message": "聊天会话已成功取消",
    "timestamp": 1234567890.123
}
```

**错误响应**:
```json
{
    "detail": "会话 session_id 不存在或已结束"
}
```

### 3. 会话状态查询接口

**端点**: `GET /api/v1/chat/session/{session_id}`

**响应格式**:
```json
{
    "session_id": "your_session_id",
    "status": "active",
    "created_at": "2025-01-01T12:00:00",
    "last_activity": "2025-01-01T12:05:00",
    "duration": 300.5,
    "request_data": {...}
}
```

### 4. 会话列表接口

**端点**: `GET /api/v1/chat/sessions`

**响应格式**:
```json
{
    "total_sessions": 5,
    "active_sessions": 2,
    "cancelled_sessions": 1,
    "completed_sessions": 2,
    "sessions": [
        {
            "session_id": "session_1",
            "status": "active",
            "created_at": "2025-01-01T12:00:00",
            "last_activity": "2025-01-01T12:05:00",
            "duration": 300.5
        }
    ]
}
```

### 5. 健康检查接口 (增强版)

**端点**: `GET /api/v1/health`

**新增会话管理器状态**:
```json
{
    "status": "healthy",
    "version": "2.0.0",
    "timestamp": 1234567890.123,
    "services": {...},
    "session_manager": {
        "total_sessions": 5,
        "active_sessions": 2,
        "cancelled_sessions": 1,
        "completed_sessions": 2
    }
}
```

## 使用示例

### Python 客户端示例

```python
import asyncio
import aiohttp
import json

async def chat_with_cancel_example():
    session_id = "my_chat_session"
    
    async with aiohttp.ClientSession() as session:
        # 1. 启动流式聊天
        chat_data = {
            "query": "请详细介绍人工智能",
            "session_id": session_id,
            "context": {}
        }
        
        # 在后台启动聊天流
        chat_task = asyncio.create_task(
            start_chat_stream(session, chat_data)
        )
        
        # 2. 等待一段时间后取消
        await asyncio.sleep(3)
        
        # 3. 取消聊天
        cancel_data = {"session_id": session_id}
        async with session.post(
            "http://localhost:8070/api/v1/chat/cancel",
            json=cancel_data
        ) as response:
            result = await response.json()
            print(f"取消结果: {result}")
        
        # 4. 等待聊天任务完成
        try:
            await chat_task
        except asyncio.CancelledError:
            print("聊天任务已取消")

async def start_chat_stream(session, data):
    async with session.post(
        "http://localhost:8070/api/v1/chat/stream",
        json=data
    ) as response:
        async for line in response.content:
            if line:
                event_data = json.loads(line.decode())
                print(f"收到事件: {event_data}")
                
                if event_data.get("event") == "cancelled":
                    print("聊天已被取消")
                    break

# 运行示例
asyncio.run(chat_with_cancel_example())
```

### JavaScript 客户端示例

```javascript
class ChatClient {
    constructor(baseUrl = 'http://localhost:8070') {
        this.baseUrl = baseUrl;
    }
    
    async startChatStream(sessionId, query) {
        const response = await fetch(`${this.baseUrl}/api/v1/chat/stream`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                session_id: sessionId,
                context: {}
            })
        });
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const eventData = JSON.parse(line.slice(6));
                        console.log('收到事件:', eventData);
                        
                        if (eventData.event === 'cancelled') {
                            console.log('聊天已被取消');
                            return;
                        }
                    }
                }
            }
        } finally {
            reader.releaseLock();
        }
    }
    
    async cancelChat(sessionId) {
        const response = await fetch(`${this.baseUrl}/api/v1/chat/cancel`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: sessionId
            })
        });
        
        return await response.json();
    }
}

// 使用示例
async function example() {
    const client = new ChatClient();
    const sessionId = `session_${Date.now()}`;
    
    // 启动聊天流
    const chatPromise = client.startChatStream(sessionId, "介绍机器学习");
    
    // 3秒后取消
    setTimeout(async () => {
        const result = await client.cancelChat(sessionId);
        console.log('取消结果:', result);
    }, 3000);
    
    await chatPromise;
}
```

## 测试

### 运行测试脚本

```bash
# 确保算法服务正在运行
cd /Users/lintao/important/ai-customer/voicehelper/algo
python test_cancel_chat.py
```

### 测试场景

1. **基础取消功能测试**
   - 启动聊天流
   - 取消聊天
   - 验证取消状态

2. **并发会话测试**
   - 创建多个并发会话
   - 选择性取消部分会话
   - 验证其他会话不受影响

3. **边界条件测试**
   - 取消不存在的会话
   - 重复取消同一会话
   - 取消已完成的会话

## 技术特性

### 1. 线程安全
- 使用 `threading.RLock()` 保护会话数据
- 原子操作确保状态一致性

### 2. 异步支持
- 使用 `asyncio.Event` 实现异步取消信号
- 非阻塞的会话管理操作

### 3. 资源管理
- 自动清理过期会话（30分钟无活动）
- 延迟清理机制确保客户端接收完整事件

### 4. 错误处理
- 优雅处理各种异常情况
- 详细的错误日志和状态码

### 5. 监控支持
- 会话统计信息
- 健康检查集成
- 详细的操作日志

## 配置选项

### 会话超时配置
```python
# 在 ChatSessionManager.cleanup_expired_sessions() 中
SESSION_TIMEOUT = 1800  # 30分钟，单位：秒
```

### 清理任务间隔
```python
# 在 start_session_cleanup_task() 中
CLEANUP_INTERVAL = 300  # 5分钟，单位：秒
```

### 延迟清理时间
```python
# 在 delayed_session_cleanup() 中
CLEANUP_DELAY = 5  # 5秒，单位：秒
```

## 故障排除

### 常见问题

1. **会话不存在错误**
   - 检查 session_id 是否正确
   - 确认会话未过期
   - 查看服务日志

2. **取消不生效**
   - 确认流式处理正在进行
   - 检查网络连接
   - 验证事件循环状态

3. **内存泄漏**
   - 检查会话清理任务是否正常运行
   - 监控活跃会话数量
   - 调整清理间隔

### 日志分析

关键日志消息：
```
INFO: 创建聊天会话: session_123
INFO: 会话已取消: session_123
INFO: 聊天会话被取消: session_123
INFO: 清理过期会话: session_456
INFO: 会话已移除: session_789
```

## 性能考虑

### 内存使用
- 每个会话约占用 1-2KB 内存
- 自动清理机制防止内存泄漏
- 建议监控活跃会话数量

### CPU 使用
- 取消操作为 O(1) 复杂度
- 清理任务为 O(n) 复杂度，n 为会话数
- 异步操作避免阻塞主线程

### 网络影响
- 取消信号通过现有连接发送
- 最小化额外网络开销
- 支持批量操作优化

## 未来改进

1. **持久化支持**
   - 会话状态持久化到数据库
   - 服务重启后恢复会话

2. **分布式支持**
   - 跨服务实例的会话管理
   - Redis 集群支持

3. **更细粒度控制**
   - 部分取消（如只取消 TTS）
   - 暂停/恢复功能

4. **监控增强**
   - Prometheus 指标集成
   - 实时监控面板

## 总结

聊天取消功能为 VoiceHelper 提供了重要的用户体验改进和资源管理能力。通过完善的会话管理机制、异步事件处理和自动清理功能，确保了系统的稳定性和可扩展性。

该功能已完全实现并通过测试，可以立即投入生产使用。
