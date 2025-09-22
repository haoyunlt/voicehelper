# VoiceHelper API 使用指南

## 📋 目录

- [API 概述](#api-概述)
- [认证方式](#认证方式)
- [请求格式](#请求格式)
- [响应格式](#响应格式)
- [错误处理](#错误处理)
- [核心 API](#核心-api)
- [SDK 使用](#sdk-使用)
- [示例代码](#示例代码)
- [最佳实践](#最佳实践)
- [限流和配额](#限流和配额)

## 🎯 API 概述

VoiceHelper 提供 RESTful API 接口，支持智能对话、语音处理、知识管理等核心功能。

### API 基础信息

| 项目 | 信息 |
|------|------|
| **Base URL** | `https://api.voicehelper.com/api/v1` |
| **协议** | HTTPS |
| **格式** | JSON |
| **编码** | UTF-8 |
| **版本** | v1.20.0 |

### API 架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client App    │    │   API Gateway   │    │   Backend       │
│   (Your App)    │◄──►│   (Rate Limit)  │◄──►│   Services      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       ▼
         │                       │              ┌─────────────────┐
         │                       │              │   Algorithm     │
         │                       │              │   Service       │
         │                       │              └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   WebSocket     │    │   Database      │    │   Vector DB     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🔐 认证方式

### API Key 认证

**获取 API Key**:
1. 登录 VoiceHelper 控制台
2. 进入 "API 管理" 页面
3. 点击 "生成新的 API Key"
4. 复制并保存 API Key

**使用方式**:
```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.voicehelper.com/api/v1/chat
```

### JWT Token 认证

**获取 Token**:
```bash
curl -X POST https://api.voicehelper.com/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your_username",
    "password": "your_password"
  }'
```

**响应**:
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600,
  "user_id": "user123"
}
```

**使用 Token**:
```bash
curl -H "Authorization: Bearer JWT_TOKEN" \
     https://api.voicehelper.com/api/v1/conversations
```

## 📝 请求格式

### HTTP 方法

| 方法 | 用途 | 示例 |
|------|------|------|
| **GET** | 获取资源 | 获取对话历史 |
| **POST** | 创建资源 | 发送消息 |
| **PUT** | 更新资源 | 更新用户信息 |
| **DELETE** | 删除资源 | 删除对话 |

### 请求头

```http
Content-Type: application/json
Authorization: Bearer YOUR_API_KEY
User-Agent: YourApp/1.0
Accept: application/json
```

### 请求体

```json
{
  "message": "你好，我想了解产品功能",
  "user_id": "user123",
  "conversation_id": "conv456",
  "metadata": {
    "source": "web",
    "timestamp": "2025-01-21T10:00:00Z"
  }
}
```

## 📤 响应格式

### 成功响应

```json
{
  "success": true,
  "data": {
    "message_id": "msg789",
    "reply": "VoiceHelper 是一个智能聊天机器人平台...",
    "confidence": 0.95,
    "processing_time": 1.2
  },
  "timestamp": "2025-01-21T10:00:01Z"
}
```

### 分页响应

```json
{
  "success": true,
  "data": {
    "items": [...],
    "pagination": {
      "page": 1,
      "limit": 20,
      "total": 100,
      "has_next": true,
      "has_prev": false
    }
  }
}
```

### 错误响应

```json
{
  "success": false,
  "error": {
    "code": "INVALID_REQUEST",
    "message": "请求参数无效",
    "details": {
      "field": "user_id",
      "reason": "用户ID不能为空"
    }
  },
  "timestamp": "2025-01-21T10:00:01Z"
}
```

## ❌ 错误处理

### HTTP 状态码

| 状态码 | 含义 | 说明 |
|--------|------|------|
| **200** | OK | 请求成功 |
| **201** | Created | 资源创建成功 |
| **400** | Bad Request | 请求参数错误 |
| **401** | Unauthorized | 认证失败 |
| **403** | Forbidden | 权限不足 |
| **404** | Not Found | 资源不存在 |
| **429** | Too Many Requests | 请求过于频繁 |
| **500** | Internal Server Error | 服务器内部错误 |

### 错误代码

| 错误代码 | 说明 | 解决方案 |
|----------|------|----------|
| `INVALID_API_KEY` | API Key 无效 | 检查 API Key 是否正确 |
| `RATE_LIMIT_EXCEEDED` | 超出限流 | 减少请求频率 |
| `INSUFFICIENT_QUOTA` | 配额不足 | 升级套餐或联系客服 |
| `INVALID_REQUEST` | 请求参数错误 | 检查请求参数格式 |
| `RESOURCE_NOT_FOUND` | 资源不存在 | 确认资源ID是否正确 |
| `SERVICE_UNAVAILABLE` | 服务不可用 | 稍后重试或联系技术支持 |

### 错误处理示例

```javascript
async function handleApiCall() {
  try {
    const response = await fetch('/api/v1/chat', {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + apiKey,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        message: 'Hello',
        user_id: 'user123'
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(`API Error: ${error.error.code} - ${error.error.message}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error('API调用失败:', error.message);
    // 根据错误类型进行处理
    if (error.message.includes('RATE_LIMIT_EXCEEDED')) {
      // 等待后重试
      setTimeout(() => handleApiCall(), 60000);
    }
  }
}
```

## 🔌 核心 API

### 1. 对话管理 API

#### 创建对话

```http
POST /api/v1/conversations
```

**请求参数**:
```json
{
  "user_id": "user123",
  "channel": "web",
  "context": {
    "source": "website",
    "page": "product",
    "user_agent": "Mozilla/5.0..."
  }
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "conversation_id": "conv_abc123",
    "user_id": "user123",
    "channel": "web",
    "status": "active",
    "created_at": "2025-01-21T10:00:00Z"
  }
}
```

#### 发送消息

```http
POST /api/v1/conversations/{conversation_id}/messages
```

**请求参数**:
```json
{
  "message": "你好，我想了解产品功能",
  "message_type": "text",
  "metadata": {
    "timestamp": "2025-01-21T10:00:00Z",
    "client_ip": "192.168.1.1"
  }
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "message_id": "msg_xyz789",
    "reply": "您好！VoiceHelper 是一个智能聊天机器人平台，主要功能包括：\n1. 自然语言对话\n2. 语音识别和合成\n3. 知识库问答\n4. 多渠道接入\n\n请问您想了解哪个方面的详细信息？",
    "confidence": 0.95,
    "intent": "product_inquiry",
    "entities": [
      {
        "type": "product",
        "value": "功能",
        "confidence": 0.9
      }
    ],
    "processing_time": 1.2,
    "created_at": "2025-01-21T10:00:01Z"
  }
}
```

#### 获取对话历史

```http
GET /api/v1/conversations/{conversation_id}/messages
```

**查询参数**:
- `limit`: 返回消息数量 (默认: 20, 最大: 100)
- `offset`: 偏移量 (默认: 0)
- `order`: 排序方式 (asc/desc, 默认: desc)

**响应**:
```json
{
  "success": true,
  "data": {
    "messages": [
      {
        "message_id": "msg_001",
        "content": "你好",
        "sender": "user",
        "created_at": "2025-01-21T10:00:00Z"
      },
      {
        "message_id": "msg_002",
        "content": "您好！有什么可以帮助您的吗？",
        "sender": "assistant",
        "created_at": "2025-01-21T10:00:01Z"
      }
    ],
    "pagination": {
      "limit": 20,
      "offset": 0,
      "total": 2,
      "has_next": false
    }
  }
}
```

### 2. 语音处理 API

#### 语音识别 (ASR)

```http
POST /api/v1/voice/asr
```

**请求格式**: `multipart/form-data`

**参数**:
- `audio`: 音频文件 (支持 wav, mp3, webm)
- `language`: 语言代码 (zh-CN, en-US 等)
- `sample_rate`: 采样率 (可选, 默认: 16000)

**cURL 示例**:
```bash
curl -X POST https://api.voicehelper.com/api/v1/voice/asr \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "audio=@voice.wav" \
  -F "language=zh-CN"
```

**响应**:
```json
{
  "success": true,
  "data": {
    "transcript": "你好，我想了解一下产品功能",
    "confidence": 0.95,
    "language": "zh-CN",
    "duration": 3.2,
    "processing_time": 0.8,
    "words": [
      {
        "word": "你好",
        "start_time": 0.0,
        "end_time": 0.5,
        "confidence": 0.98
      },
      {
        "word": "我",
        "start_time": 0.6,
        "end_time": 0.8,
        "confidence": 0.95
      }
    ]
  }
}
```

#### 语音合成 (TTS)

```http
POST /api/v1/voice/tts
```

**请求参数**:
```json
{
  "text": "欢迎使用VoiceHelper智能助手",
  "voice": "zh-CN-XiaoxiaoNeural",
  "speed": 1.0,
  "pitch": 0,
  "volume": 1.0,
  "format": "wav"
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "audio_url": "https://api.voicehelper.com/audio/tts_abc123.wav",
    "duration": 2.5,
    "format": "wav",
    "size": 80000,
    "expires_at": "2025-01-21T11:00:00Z"
  }
}
```

### 3. 知识管理 API

#### 上传文档

```http
POST /api/v1/documents
```

**请求格式**: `multipart/form-data`

**参数**:
- `file`: 文档文件 (支持 pdf, txt, docx, md)
- `title`: 文档标题
- `category`: 文档分类
- `tags`: 标签 (可选)

**cURL 示例**:
```bash
curl -X POST https://api.voicehelper.com/api/v1/documents \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@manual.pdf" \
  -F "title=产品使用手册" \
  -F "category=documentation" \
  -F "tags=manual,guide"
```

**响应**:
```json
{
  "success": true,
  "data": {
    "document_id": "doc_abc123",
    "title": "产品使用手册",
    "category": "documentation",
    "file_size": 1024000,
    "pages": 50,
    "status": "processing",
    "created_at": "2025-01-21T10:00:00Z",
    "estimated_completion": "2025-01-21T10:05:00Z"
  }
}
```

#### 知识搜索

```http
POST /api/v1/rag/search
```

**请求参数**:
```json
{
  "query": "如何重置密码？",
  "top_k": 5,
  "filters": {
    "category": "faq",
    "tags": ["account", "security"]
  },
  "similarity_threshold": 0.7
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "results": [
      {
        "document_id": "doc_001",
        "title": "账户管理FAQ",
        "content": "重置密码的步骤：1. 点击登录页面的'忘记密码'链接...",
        "similarity": 0.92,
        "metadata": {
          "category": "faq",
          "page": 5,
          "section": "密码管理"
        }
      }
    ],
    "total_results": 1,
    "processing_time": 0.3
  }
}
```

#### RAG 问答

```http
POST /api/v1/rag/query
```

**请求参数**:
```json
{
  "query": "请详细说明如何重置密码",
  "top_k": 5,
  "include_sources": true,
  "max_tokens": 500
}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "answer": "重置密码的详细步骤如下：\n\n1. 访问登录页面，点击"忘记密码"链接\n2. 输入您的注册邮箱地址\n3. 检查邮箱中的重置链接\n4. 点击链接并设置新密码\n5. 使用新密码登录系统\n\n注意：重置链接有效期为24小时，请及时使用。",
    "confidence": 0.88,
    "sources": [
      {
        "document_id": "doc_001",
        "title": "账户管理FAQ",
        "relevance": 0.92
      }
    ],
    "processing_time": 2.1
  }
}
```

### 4. 用户管理 API

#### 创建用户

```http
POST /api/v1/users
```

**请求参数**:
```json
{
  "user_id": "user123",
  "name": "张三",
  "email": "zhangsan@example.com",
  "preferences": {
    "language": "zh-CN",
    "voice_enabled": true,
    "theme": "light"
  }
}
```

#### 获取用户信息

```http
GET /api/v1/users/{user_id}
```

**响应**:
```json
{
  "success": true,
  "data": {
    "user_id": "user123",
    "name": "张三",
    "email": "zhangsan@example.com",
    "preferences": {
      "language": "zh-CN",
      "voice_enabled": true,
      "theme": "light"
    },
    "statistics": {
      "total_conversations": 25,
      "total_messages": 150,
      "last_active": "2025-01-21T09:30:00Z"
    },
    "created_at": "2025-01-01T00:00:00Z"
  }
}
```

### 5. 分析统计 API

#### 对话统计

```http
GET /api/v1/analytics/conversations
```

**查询参数**:
- `start_date`: 开始日期 (YYYY-MM-DD)
- `end_date`: 结束日期 (YYYY-MM-DD)
- `period`: 时间周期 (1d, 7d, 30d)
- `group_by`: 分组方式 (day, hour, channel)

**响应**:
```json
{
  "success": true,
  "data": {
    "total_conversations": 1250,
    "total_messages": 8500,
    "avg_messages_per_conversation": 6.8,
    "satisfaction_score": 4.2,
    "breakdown": [
      {
        "date": "2025-01-21",
        "conversations": 85,
        "messages": 578,
        "avg_response_time": 1.2
      }
    ]
  }
}
```

#### 用户分析

```http
GET /api/v1/analytics/users
```

**响应**:
```json
{
  "success": true,
  "data": {
    "total_users": 5000,
    "active_users": 1200,
    "new_users": 150,
    "retention_rate": 0.75,
    "top_channels": [
      {
        "channel": "web",
        "users": 800,
        "percentage": 66.7
      },
      {
        "channel": "mobile",
        "users": 300,
        "percentage": 25.0
      }
    ]
  }
}
```

## 📦 SDK 使用

### JavaScript SDK

**安装**:
```bash
npm install @voicehelper/sdk
```

**使用示例**:
```javascript
import VoiceHelper from '@voicehelper/sdk';

const client = new VoiceHelper({
  apiKey: 'your-api-key',
  baseURL: 'https://api.voicehelper.com'
});

// 发送消息
async function sendMessage() {
  try {
    const response = await client.chat.send({
      message: '你好，我想了解产品功能',
      userId: 'user123'
    });
    
    console.log('AI回复:', response.reply);
  } catch (error) {
    console.error('发送失败:', error.message);
  }
}

// 语音识别
async function recognizeVoice(audioBlob) {
  try {
    const result = await client.voice.recognize(audioBlob, {
      language: 'zh-CN'
    });
    
    console.log('识别结果:', result.transcript);
  } catch (error) {
    console.error('识别失败:', error.message);
  }
}

// 知识搜索
async function searchKnowledge() {
  try {
    const results = await client.knowledge.search({
      query: '如何使用语音功能？',
      topK: 3
    });
    
    results.forEach(result => {
      console.log(`相关度: ${result.similarity}`);
      console.log(`内容: ${result.content}`);
    });
  } catch (error) {
    console.error('搜索失败:', error.message);
  }
}
```

### Python SDK

**安装**:
```bash
pip install voicehelper-sdk
```

**使用示例**:
```python
from voicehelper import VoiceHelperClient

client = VoiceHelperClient(
    api_key='your-api-key',
    base_url='https://api.voicehelper.com'
)

# 发送消息
def send_message():
    try:
        response = client.chat.send(
            message='你好，我想了解产品功能',
            user_id='user123'
        )
        print(f'AI回复: {response.reply}')
    except Exception as e:
        print(f'发送失败: {e}')

# 上传文档
def upload_document():
    try:
        with open('document.pdf', 'rb') as f:
            result = client.documents.upload(
                file=f,
                title='产品手册',
                category='documentation'
            )
        print(f'文档ID: {result.document_id}')
    except Exception as e:
        print(f'上传失败: {e}')

# RAG 问答
def rag_query():
    try:
        response = client.rag.query(
            query='如何重置密码？',
            top_k=5,
            include_sources=True
        )
        print(f'答案: {response.answer}')
        print(f'来源: {response.sources}')
    except Exception as e:
        print(f'查询失败: {e}')
```

### Go SDK

**安装**:
```bash
go get github.com/voicehelper/go-sdk
```

**使用示例**:
```go
package main

import (
    "context"
    "fmt"
    "log"
    
    "github.com/voicehelper/go-sdk"
)

func main() {
    client := voicehelper.NewClient(&voicehelper.Config{
        APIKey:  "your-api-key",
        BaseURL: "https://api.voicehelper.com",
    })

    // 发送消息
    response, err := client.Chat.Send(context.Background(), &voicehelper.ChatRequest{
        Message: "你好，我想了解产品功能",
        UserID:  "user123",
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("AI回复: %s\n", response.Reply)

    // 创建对话
    conv, err := client.Conversations.Create(context.Background(), &voicehelper.ConversationRequest{
        UserID:  "user123",
        Channel: "api",
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Printf("对话ID: %s\n", conv.ConversationID)
}
```

## 💡 示例代码

### 完整聊天应用示例

**HTML + JavaScript**:
```html
<!DOCTYPE html>
<html>
<head>
    <title>VoiceHelper 聊天示例</title>
    <style>
        .chat-container { max-width: 600px; margin: 0 auto; padding: 20px; }
        .messages { height: 400px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; }
        .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
        .user { background: #007bff; color: white; text-align: right; }
        .assistant { background: #f8f9fa; }
        .input-area { margin-top: 10px; display: flex; }
        .input-area input { flex: 1; padding: 10px; }
        .input-area button { padding: 10px 20px; margin-left: 10px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div id="messages" class="messages"></div>
        <div class="input-area">
            <input type="text" id="messageInput" placeholder="输入消息..." />
            <button onclick="sendMessage()">发送</button>
            <button onclick="startVoiceRecognition()">🎤</button>
        </div>
    </div>

    <script>
        const API_KEY = 'your-api-key';
        const BASE_URL = 'https://api.voicehelper.com/api/v1';
        let conversationId = null;

        // 初始化对话
        async function initConversation() {
            try {
                const response = await fetch(`${BASE_URL}/conversations`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${API_KEY}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        user_id: 'demo_user',
                        channel: 'web'
                    })
                });
                
                const data = await response.json();
                conversationId = data.data.conversation_id;
            } catch (error) {
                console.error('初始化对话失败:', error);
            }
        }

        // 发送消息
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // 显示用户消息
            addMessage(message, 'user');
            input.value = '';
            
            try {
                const response = await fetch(`${BASE_URL}/conversations/${conversationId}/messages`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${API_KEY}`,
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        message: message,
                        message_type: 'text'
                    })
                });
                
                const data = await response.json();
                
                // 显示AI回复
                addMessage(data.data.reply, 'assistant');
            } catch (error) {
                console.error('发送消息失败:', error);
                addMessage('抱歉，发送失败，请重试。', 'assistant');
            }
        }

        // 添加消息到界面
        function addMessage(content, sender) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            messageDiv.textContent = content;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        // 语音识别
        function startVoiceRecognition() {
            if ('webkitSpeechRecognition' in window) {
                const recognition = new webkitSpeechRecognition();
                recognition.lang = 'zh-CN';
                recognition.continuous = false;
                recognition.interimResults = false;

                recognition.onresult = function(event) {
                    const transcript = event.results[0][0].transcript;
                    document.getElementById('messageInput').value = transcript;
                };

                recognition.onerror = function(event) {
                    console.error('语音识别错误:', event.error);
                };

                recognition.start();
            } else {
                alert('您的浏览器不支持语音识别功能');
            }
        }

        // 回车发送
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // 初始化
        initConversation();
    </script>
</body>
</html>
```

### React 聊天组件

```jsx
import React, { useState, useEffect, useRef } from 'react';

const VoiceChatComponent = ({ apiKey, userId }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [conversationId, setConversationId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const BASE_URL = 'https://api.voicehelper.com/api/v1';

  // 初始化对话
  useEffect(() => {
    initConversation();
  }, []);

  // 自动滚动到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const initConversation = async () => {
    try {
      const response = await fetch(`${BASE_URL}/conversations`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_id: userId,
          channel: 'web'
        })
      });
      
      const data = await response.json();
      setConversationId(data.data.conversation_id);
    } catch (error) {
      console.error('初始化对话失败:', error);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || !conversationId || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    setIsLoading(true);

    // 添加用户消息
    setMessages(prev => [...prev, {
      id: Date.now(),
      content: userMessage,
      sender: 'user',
      timestamp: new Date()
    }]);

    try {
      const response = await fetch(`${BASE_URL}/conversations/${conversationId}/messages`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: userMessage,
          message_type: 'text'
        })
      });

      const data = await response.json();

      // 添加AI回复
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        content: data.data.reply,
        sender: 'assistant',
        timestamp: new Date(),
        confidence: data.data.confidence
      }]);
    } catch (error) {
      console.error('发送消息失败:', error);
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        content: '抱歉，发送失败，请重试。',
        sender: 'assistant',
        timestamp: new Date()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map(message => (
          <div key={message.id} className={`message ${message.sender}`}>
            <div className="content">{message.content}</div>
            <div className="timestamp">
              {message.timestamp.toLocaleTimeString()}
              {message.confidence && (
                <span className="confidence">
                  置信度: {(message.confidence * 100).toFixed(1)}%
                </span>
              )}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message assistant loading">
            <div className="content">正在思考中...</div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <div className="input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="输入消息..."
          disabled={isLoading}
        />
        <button onClick={sendMessage} disabled={!input.trim() || isLoading}>
          发送
        </button>
      </div>
    </div>
  );
};

export default VoiceChatComponent;
```

## 🎯 最佳实践

### 1. 错误处理和重试

```javascript
class VoiceHelperClient {
  constructor(apiKey, options = {}) {
    this.apiKey = apiKey;
    this.baseURL = options.baseURL || 'https://api.voicehelper.com/api/v1';
    this.maxRetries = options.maxRetries || 3;
    this.retryDelay = options.retryDelay || 1000;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Authorization': `Bearer ${this.apiKey}`,
        'Content-Type': 'application/json',
        ...options.headers
      },
      ...options
    };

    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        const response = await fetch(url, config);
        
        if (!response.ok) {
          const error = await response.json();
          throw new APIError(error.error.code, error.error.message, response.status);
        }
        
        return await response.json();
      } catch (error) {
        if (attempt === this.maxRetries || !this.shouldRetry(error)) {
          throw error;
        }
        
        await this.delay(this.retryDelay * attempt);
      }
    }
  }

  shouldRetry(error) {
    // 5xx 错误或网络错误可以重试
    return error.status >= 500 || error.code === 'NETWORK_ERROR';
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

class APIError extends Error {
  constructor(code, message, status) {
    super(message);
    this.code = code;
    this.status = status;
    this.name = 'APIError';
  }
}
```

### 2. 请求缓存

```javascript
class CacheManager {
  constructor(ttl = 300000) { // 5分钟缓存
    this.cache = new Map();
    this.ttl = ttl;
  }

  get(key) {
    const item = this.cache.get(key);
    if (!item) return null;
    
    if (Date.now() > item.expiry) {
      this.cache.delete(key);
      return null;
    }
    
    return item.data;
  }

  set(key, data) {
    this.cache.set(key, {
      data,
      expiry: Date.now() + this.ttl
    });
  }

  clear() {
    this.cache.clear();
  }
}

// 使用缓存的客户端
class CachedVoiceHelperClient extends VoiceHelperClient {
  constructor(apiKey, options = {}) {
    super(apiKey, options);
    this.cache = new CacheManager(options.cacheTTL);
  }

  async searchKnowledge(query, options = {}) {
    const cacheKey = `search:${JSON.stringify({ query, ...options })}`;
    const cached = this.cache.get(cacheKey);
    
    if (cached) {
      return cached;
    }
    
    const result = await this.request('/rag/search', {
      method: 'POST',
      body: JSON.stringify({ query, ...options })
    });
    
    this.cache.set(cacheKey, result);
    return result;
  }
}
```

### 3. 批量处理

```javascript
class BatchProcessor {
  constructor(client, options = {}) {
    this.client = client;
    this.batchSize = options.batchSize || 10;
    this.delay = options.delay || 100;
  }

  async processDocuments(documents) {
    const results = [];
    
    for (let i = 0; i < documents.length; i += this.batchSize) {
      const batch = documents.slice(i, i + this.batchSize);
      
      const batchPromises = batch.map(doc => 
        this.client.documents.upload(doc).catch(error => ({
          error: error.message,
          document: doc
        }))
      );
      
      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);
      
      // 避免过于频繁的请求
      if (i + this.batchSize < documents.length) {
        await new Promise(resolve => setTimeout(resolve, this.delay));
      }
    }
    
    return results;
  }
}
```

### 4. WebSocket 实时通信

```javascript
class VoiceHelperWebSocket {
  constructor(apiKey, options = {}) {
    this.apiKey = apiKey;
    this.wsURL = options.wsURL || 'wss://api.voicehelper.com/ws';
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = options.maxReconnectAttempts || 5;
    this.reconnectDelay = options.reconnectDelay || 1000;
    this.eventHandlers = new Map();
  }

  connect() {
    this.ws = new WebSocket(`${this.wsURL}?token=${this.apiKey}`);
    
    this.ws.onopen = () => {
      console.log('WebSocket 连接已建立');
      this.reconnectAttempts = 0;
      this.emit('connected');
    };
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.emit(data.type, data.payload);
    };
    
    this.ws.onclose = () => {
      console.log('WebSocket 连接已关闭');
      this.emit('disconnected');
      this.reconnect();
    };
    
    this.ws.onerror = (error) => {
      console.error('WebSocket 错误:', error);
      this.emit('error', error);
    };
  }

  reconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`尝试重连 (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        this.connect();
      }, this.reconnectDelay * this.reconnectAttempts);
    }
  }

  send(type, payload) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, payload }));
    }
  }

  on(event, handler) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, []);
    }
    this.eventHandlers.get(event).push(handler);
  }

  emit(event, data) {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => handler(data));
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// 使用示例
const wsClient = new VoiceHelperWebSocket('your-api-key');

wsClient.on('connected', () => {
  console.log('已连接到实时服务');
});

wsClient.on('message', (data) => {
  console.log('收到实时消息:', data);
});

wsClient.connect();
```

## 🚦 限流和配额

### 限流规则

| 套餐类型 | 每分钟请求数 | 每日请求数 | 并发连接数 |
|----------|--------------|------------|------------|
| **免费版** | 60 | 1,000 | 5 |
| **基础版** | 300 | 10,000 | 20 |
| **专业版** | 1,000 | 50,000 | 100 |
| **企业版** | 5,000 | 无限制 | 500 |

### 限流响应

当超出限流时，API 会返回 `429 Too Many Requests` 状态码：

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "请求过于频繁，请稍后重试",
    "details": {
      "limit": 60,
      "remaining": 0,
      "reset_time": "2025-01-21T10:01:00Z"
    }
  }
}
```

### 限流处理策略

```javascript
class RateLimitHandler {
  constructor(client) {
    this.client = client;
    this.requestQueue = [];
    this.isProcessing = false;
  }

  async request(endpoint, options) {
    return new Promise((resolve, reject) => {
      this.requestQueue.push({ endpoint, options, resolve, reject });
      this.processQueue();
    });
  }

  async processQueue() {
    if (this.isProcessing || this.requestQueue.length === 0) {
      return;
    }

    this.isProcessing = true;

    while (this.requestQueue.length > 0) {
      const { endpoint, options, resolve, reject } = this.requestQueue.shift();

      try {
        const result = await this.client.request(endpoint, options);
        resolve(result);
      } catch (error) {
        if (error.status === 429) {
          // 重新加入队列
          this.requestQueue.unshift({ endpoint, options, resolve, reject });
          
          // 等待重置时间
          const resetTime = new Date(error.details?.reset_time || Date.now() + 60000);
          const waitTime = resetTime.getTime() - Date.now();
          
          console.log(`触发限流，等待 ${waitTime}ms 后重试`);
          await new Promise(resolve => setTimeout(resolve, waitTime));
        } else {
          reject(error);
        }
      }
    }

    this.isProcessing = false;
  }
}
```

---

## 📞 技术支持

如果在使用 API 过程中遇到问题，可以通过以下方式获取帮助：

- **API 文档**: [https://docs.voicehelper.com/api](https://docs.voicehelper.com/api)
- **SDK 文档**: [https://docs.voicehelper.com/sdk](https://docs.voicehelper.com/sdk)
- **GitHub Issues**: [问题反馈](https://github.com/your-org/voicehelper/issues)
- **技术支持**: api-support@voicehelper.com
- **开发者社区**: [Discord](https://discord.gg/voicehelper)

---

**API 使用指南完成！** 🎉

现在你已经掌握了 VoiceHelper API 的完整使用方法，可以开始构建你的智能对话应用了！
