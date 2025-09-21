# 🛠️ 开发指南 - VoiceHelper AI

## 📋 概述

本指南提供了VoiceHelper AI智能对话系统的完整开发指南，包括SDK使用、测试数据集、小程序开发等内容。

---

## 🚀 SDK 使用指南

### 快速开始

#### JavaScript/TypeScript SDK

```bash
npm install @voicehelper/sdk
```

```typescript
import { VoiceHelperSDK } from '@voicehelper/sdk';

const client = new VoiceHelperSDK({
  apiKey: 'your-api-key',
  baseURL: 'https://api.voicehelper.ai/v1'
});

// 智能对话
const response = await client.createChatCompletion({
  messages: [{ role: 'user', content: '你好！' }],
  model: 'gpt-4'
});

// 语音转文字
const transcription = await client.transcribeAudio({
  file: audioFile,
  model: 'whisper-1'
});
```

#### Python SDK

```bash
pip install voicehelper-sdk
```

```python
from voicehelper_sdk import VoiceHelperSDK, VoiceHelperConfig

config = VoiceHelperConfig(api_key="your-api-key")
client = VoiceHelperSDK(config)

# 智能对话
response = await client.create_chat_completion({
    "messages": [{"role": "user", "content": "你好！"}],
    "model": "gpt-4"
})

# 语音转文字
with open("audio.wav", "rb") as audio_file:
    transcription = await client.transcribe_audio(audio_file)
```

### 高级功能

#### 1. 流式对话

```typescript
// JavaScript流式响应
await client.messages.sendStream(
  conversationId,
  { content: '请解释人工智能', stream: true },
  (event) => {
    switch (event.type) {
      case 'delta':
        process.stdout.write(event.content || '');
        break;
      case 'done':
        console.log(`\n消息完成，ID: ${event.message_id}`);
        break;
      case 'error':
        console.error('流式错误:', event.error);
        break;
    }
  }
);
```

```python
# Python流式响应
async for event in client.messages.send_stream(
    conversation_id,
    SendMessageRequest(content="请解释人工智能")
):
    if event.type == "delta":
        print(event.content, end="", flush=True)
    elif event.type == "done":
        print(f"\n消息ID: {event.message_id}")
```

#### 2. 多模态处理

```typescript
// 语音消息处理
class MultiModalChat {
  async sendVoiceMessage(audioBlob: Blob): Promise<void> {
    // 语音转文字
    const transcription = await this.client.voice.transcribe({
      audio: audioBlob,
      language: 'zh-CN'
    });

    // 发送文字消息获取回复
    let responseText = '';
    await this.client.messages.sendStream(
      this.conversationId,
      { content: transcription.text, modality: 'voice' },
      (event) => {
        if (event.type === 'delta') {
          responseText += event.content || '';
        } else if (event.type === 'done') {
          this.synthesizeAndPlayResponse(responseText);
        }
      }
    );
  }

  private async synthesizeAndPlayResponse(text: string): Promise<void> {
    const audioBuffer = await this.client.voice.synthesize({
      text: text,
      voice: 'female',
      speed: 1.0
    });

    const audioBlob = new Blob([audioBuffer], { type: 'audio/wav' });
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);
    audio.play();
  }
}
```

#### 3. 错误处理和重试

```typescript
import { APIError, NetworkError, ValidationError } from '@voicehelper/sdk';

class RobustClient {
  async sendMessageWithRetry(conversationId: string, content: string): Promise<string> {
    const maxRetries = 3;
    const baseDelay = 1000;

    for (let attempt = 0; attempt < maxRetries; attempt++) {
      try {
        let response = '';
        await this.client.messages.sendStream(
          conversationId,
          { content },
          (event) => {
            if (event.type === 'delta') {
              response += event.content || '';
            }
          }
        );
        return response;
      } catch (error) {
        if (error instanceof APIError && error.isRateLimitError) {
          const delay = baseDelay * Math.pow(2, attempt);
          console.log(`触发限流，等待 ${delay}ms 后重试...`);
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }
        
        if (attempt === maxRetries - 1) {
          throw error;
        }
        
        console.log(`请求失败，重试中... (${attempt + 1}/${maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, baseDelay));
      }
    }
    
    throw new Error('达到最大重试次数');
  }
}
```

---

## 🧪 测试数据集

### 数据集结构

```
tests/datasets/
├── chat/                    # 聊天对话测试数据集
│   ├── conversation_scenarios.json      # 多轮对话场景
│   ├── intent_classification.json       # 意图识别测试
│   └── emotion_analysis.json           # 情感分析测试
├── voice/                   # 语音交互测试数据集
│   ├── asr_test_cases.json             # 语音识别测试
│   ├── tts_test_cases.json             # 语音合成测试
│   └── voice_emotion_test.json         # 语音情感测试
├── rag/                     # RAG检索测试数据集
│   ├── knowledge_base_samples.json     # 知识库样本
│   └── vector_search_test.json         # 向量检索测试
├── agent/                   # 智能代理测试数据集
│   ├── tool_calling_test.json          # 工具调用测试
│   └── reasoning_chain_test.json       # 推理链测试
├── performance/             # 性能测试数据集
├── security/                # 安全测试数据集
└── integration/             # 集成测试数据集
```

### 测试数据集分类

#### 1. 聊天对话测试 (Chat)

**目标**: 验证聊天机器人的对话能力、意图识别和情感分析

- **conversation_scenarios.json**: 50个多轮对话场景
  - 产品咨询、客户投诉、技术支持等
  - 上下文理解、情感识别、异常处理
  
- **intent_classification.json**: 200个意图分类样本
  - 15种意图类型：问候、咨询、投诉、预订等
  - 包含挑战性案例和模糊表达
  
- **emotion_analysis.json**: 150个情感分析样本
  - 8种情感类型：开心、愤怒、悲伤、焦虑等
  - 复合情感和文化语境测试

#### 2. 语音交互测试 (Voice)

**目标**: 验证语音识别、语音合成和语音情感的准确性

- **asr_test_cases.json**: 100个语音识别测试
  - 清晰语音、噪音环境、口音语音等
  - 技术术语、快速语音、情感语音
  
- **tts_test_cases.json**: 80个语音合成测试
  - 基础句子、情感语音、技术内容
  - 长文本、特殊字符处理
  
- **voice_emotion_test.json**: 120个语音情感测试
  - 情感识别和情感合成
  - 跨文化语境和复合情感

#### 3. RAG检索测试 (RAG)

**目标**: 验证知识检索和文档问答的准确性

- **knowledge_base_samples.json**: 200个文档样本
  - 产品文档、技术规格、FAQ、政策等
  - 复杂查询和边界情况测试
  
- **vector_search_test.json**: 1000个向量检索测试
  - 语义相似度、跨领域查询、多语言
  - 性能测试和鲁棒性验证

### 使用测试数据

```bash
# 运行聊天对话测试
python -m pytest tests/chat/ -v

# 运行性能测试
python -m pytest tests/performance/ -v

# 运行安全测试
python -m pytest tests/security/ -v
```

```python
import json

# 加载对话测试数据
with open('tests/datasets/chat/conversation_scenarios.json') as f:
    chat_data = json.load(f)

# 获取测试场景
scenarios = chat_data['scenarios']
for scenario in scenarios:
    print(f"测试场景: {scenario['title']}")
```

---

## 📱 微信小程序开发

### 功能特性

- 🎤 **语音交互**: 支持实时语音输入和TTS语音合成
- 💬 **文本聊天**: 传统的文本消息交互
- 🔄 **流式响应**: 支持SSE和WebSocket双协议
- 🔐 **微信登录**: 集成微信授权登录
- 📱 **自适应**: 根据iOS/Android自动调整音频配置
- 🔌 **断线重连**: 自动处理网络中断和重连

### 技术架构

#### 核心文件

- `app.js` - 小程序主入口，全局状态管理
- `pages/chat/chat.js` - 聊天页面，核心交互逻辑
- `app.json` - 小程序配置文件

#### 音频处理

- **录音**: 使用 `wx.getRecorderManager()` 
- **播放**: 使用 `wx.createInnerAudioContext()` 和 `wx.createWebAudioContext()`
- **格式**: Android使用MP3，iOS使用AAC
- **采样率**: 16kHz，单声道

#### 网络通信

- **WebSocket**: 用于实时语音流
- **HTTP SSE**: 用于文本聊天流式响应
- **断线重连**: 3秒自动重连机制

### 开发配置

```javascript
// app.js
globalData: {
  apiUrl: 'https://your-api-domain.com/api/v1',
  wsUrl: 'wss://your-api-domain.com/api/v1'
}
```

### API接口

#### WebSocket协议

**连接地址**: `/api/v1/voice/stream`

**消息类型**:
- `start` - 初始化连接
- `audio` - 发送音频数据
- `stop` - 停止录音
- `cancel` - 取消请求

#### HTTP接口

- `POST /api/v1/chat/stream` - 文本聊天
- `POST /api/v1/auth/wechat/miniprogram/login` - 微信登录
- `GET /api/v1/conversations/{id}/messages` - 获取历史消息

### 部署注意事项

1. **域名配置**: 在微信公众平台配置合法域名
2. **HTTPS**: 所有接口必须使用HTTPS
3. **WSS**: WebSocket必须使用WSS协议
4. **权限申请**: 需要申请录音权限

### 性能优化

1. **音频缓冲**: 使用队列管理音频播放
2. **消息分页**: 历史消息分页加载
3. **防抖处理**: 输入和发送添加防抖
4. **资源清理**: 页面卸载时清理定时器和连接

---

## 🔧 开发最佳实践

### 1. 连接池和资源管理

```typescript
class VoiceHelperManager {
  private static instance: VoiceHelperManager;
  private client: VoiceHelperSDK;

  private constructor(apiKey: string) {
    this.client = new VoiceHelperSDK({
      apiKey,
      timeout: 30000,
    });
  }

  static getInstance(apiKey: string): VoiceHelperManager {
    if (!VoiceHelperManager.instance) {
      VoiceHelperManager.instance = new VoiceHelperManager(apiKey);
    }
    return VoiceHelperManager.instance;
  }

  getClient(): VoiceHelperSDK {
    return this.client;
  }
}
```

### 2. 缓存和性能优化

```python
import asyncio
from typing import Dict, Optional
from datetime import datetime, timedelta
from voicehelper_sdk import VoiceHelperSDK, Conversation

class CachedVoiceHelperClient:
    def __init__(self, api_key: str):
        self.client = VoiceHelperSDK(api_key=api_key)
        self.conversation_cache: Dict[str, tuple[Conversation, datetime]] = {}
        self.cache_ttl = timedelta(minutes=30)
    
    async def get_conversation_cached(self, conversation_id: str) -> Optional[Conversation]:
        """获取缓存的对话信息"""
        if conversation_id in self.conversation_cache:
            conversation, cached_at = self.conversation_cache[conversation_id]
            if datetime.now() - cached_at < self.cache_ttl:
                return conversation
            else:
                # 缓存过期，删除
                del self.conversation_cache[conversation_id]
        
        # 从API获取并缓存
        try:
            conversation = await self.client.conversations.get(conversation_id)
            self.conversation_cache[conversation_id] = (conversation, datetime.now())
            return conversation
        except Exception:
            return None
```

### 3. 监控和日志

```typescript
class MonitoredVoiceHelperClient {
  private client: VoiceHelperSDK;
  private metrics = {
    requests: 0,
    errors: 0,
    latency: [] as number[]
  };

  async sendMessage(conversationId: string, content: string): Promise<string> {
    const startTime = Date.now();
    this.metrics.requests++;

    try {
      let response = '';
      await this.client.messages.sendStream(
        conversationId,
        { content },
        (event) => {
          if (event.type === 'delta') {
            response += event.content || '';
          }
        }
      );

      const latency = Date.now() - startTime;
      this.metrics.latency.push(latency);
      
      console.log(`请求成功 - 延迟: ${latency}ms`);
      return response;
    } catch (error) {
      this.metrics.errors++;
      console.error(`请求失败: ${error.message}`);
      throw error;
    }
  }

  getMetrics() {
    const avgLatency = this.metrics.latency.length > 0
      ? this.metrics.latency.reduce((a, b) => a + b, 0) / this.metrics.latency.length
      : 0;

    return {
      totalRequests: this.metrics.requests,
      totalErrors: this.metrics.errors,
      errorRate: this.metrics.requests > 0 ? this.metrics.errors / this.metrics.requests : 0,
      averageLatency: avgLatency
    };
  }
}
```

---

## 📊 评估指标

### 功能性指标
- **准确率**: 正确结果 / 总测试数
- **召回率**: 找到的相关结果 / 总相关结果
- **F1分数**: 精确率和召回率的调和平均

### 性能指标
- **响应时间**: P50, P95, P99延迟
- **吞吐量**: 每秒处理请求数
- **并发能力**: 最大并发用户数

### 质量指标
- **可用性**: 系统正常运行时间比例
- **错误率**: 错误请求 / 总请求数
- **用户满意度**: 基于响应质量的评分

---

## 🤝 贡献指南

### 添加新功能
1. Fork项目并创建特性分支
2. 按照现有格式添加功能代码
3. 编写相应的测试用例
4. 提交Pull Request

### 报告问题
1. 使用GitHub Issues报告问题
2. 提供详细的问题描述和复现步骤
3. 包含相关的测试数据和日志

### 改进建议
1. 提出开发流程改进建议
2. 分享开发工具和方法
3. 参与代码审查和讨论

---

## 📞 技术支持

- **开发者文档**: [https://docs.voicehelper.ai](https://docs.voicehelper.ai)
- **API参考**: [https://api.voicehelper.ai/docs](https://api.voicehelper.ai/docs)
- **GitHub Issues**: [提交问题和建议](https://github.com/voicehelper/voicehelper/issues)
- **开发者社区**: [Discord](https://discord.gg/voicehelper)

---

*最后更新: 2025-09-22*  
*版本: v1.9.0*  
*维护者: VoiceHelper开发团队*
