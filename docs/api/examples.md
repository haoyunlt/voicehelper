# Chatbot SDK 使用示例

本文档提供了Chatbot SDK的详细使用示例，包括JavaScript/TypeScript和Python版本。

## 目录

- [快速开始](#快速开始)
- [JavaScript/TypeScript SDK](#javascripttypescript-sdk)
- [Python SDK](#python-sdk)
- [常见用例](#常见用例)
- [错误处理](#错误处理)
- [最佳实践](#最佳实践)

## 快速开始

### 安装

**JavaScript/TypeScript:**
```bash
npm install @chatbot/sdk
# 或
yarn add @chatbot/sdk
```

**Python:**
```bash
pip install chatbot-sdk
```

### 基本配置

**JavaScript/TypeScript:**
```typescript
import { ChatbotClient } from '@chatbot/sdk';

const client = new ChatbotClient({
  apiKey: 'your-api-key',
  baseURL: 'https://api.chatbot.ai/v1',
  tenantId: 'your-tenant-id'
});
```

**Python:**
```python
from chatbot_sdk import ChatbotClient

client = ChatbotClient(
    api_key="your-api-key",
    base_url="https://api.chatbot.ai/v1",
    tenant_id="your-tenant-id"
)
```

## JavaScript/TypeScript SDK

### 1. 用户认证

```typescript
// 微信小程序登录
try {
  const loginResponse = await client.auth.wechatLogin({
    code: 'wx_auth_code',
    nickname: '用户昵称',
    avatar: 'https://example.com/avatar.jpg'
  });
  
  console.log('登录成功:', loginResponse.user);
  console.log('访问令牌:', loginResponse.token);
  
  // 更新客户端配置
  client.updateConfig({ apiKey: loginResponse.token });
} catch (error) {
  console.error('登录失败:', error.message);
}

// 刷新令牌
try {
  const tokenResponse = await client.auth.refreshToken(refreshToken);
  client.updateConfig({ apiKey: tokenResponse.token });
} catch (error) {
  console.error('令牌刷新失败:', error.message);
}
```

### 2. 对话管理

```typescript
// 创建对话
const conversation = await client.conversations.create({
  title: '新的对话',
  metadata: { source: 'web' }
});

// 获取对话列表
const conversationList = await client.conversations.list({
  page: 1,
  limit: 20,
  status: 'active'
});

// 更新对话
const updatedConversation = await client.conversations.update(
  conversation.id,
  { title: '更新的标题' }
);

// 删除对话
await client.conversations.delete(conversation.id);
```

### 3. 消息处理

```typescript
// 发送普通消息
const message = await client.messages.send(conversation.id, {
  content: '你好，世界！',
  modality: 'text'
});

// 发送流式消息
await client.messages.sendStream(
  conversation.id,
  {
    content: '请解释一下人工智能',
    stream: true
  },
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
  },
  (error) => {
    console.error('发送失败:', error);
  }
);

// 获取消息历史
const messageHistory = await client.messages.list(conversation.id, {
  page: 1,
  limit: 50
});
```

### 4. 语音功能

```typescript
// 语音转文字
const audioFile = new File([audioBlob], 'audio.wav', { type: 'audio/wav' });
const transcription = await client.voice.transcribe({
  audio: audioFile,
  language: 'zh-CN'
});
console.log('识别结果:', transcription.text);
console.log('置信度:', transcription.confidence);

// 文字转语音
const audioBuffer = await client.voice.synthesize({
  text: '你好，这是语音合成测试',
  voice: 'female',
  speed: 1.0
});

// 播放音频
const audioBlob = new Blob([audioBuffer], { type: 'audio/wav' });
const audioUrl = URL.createObjectURL(audioBlob);
const audio = new Audio(audioUrl);
audio.play();
```

### 5. 知识库管理

```typescript
// 创建数据集
const dataset = await client.datasets.create({
  name: '产品文档',
  description: '产品相关文档集合',
  type: 'document'
});

// 上传文档
const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement;
const file = fileInput.files?.[0];
if (file) {
  const document = await client.datasets.uploadDocument(dataset.id, {
    file: file,
    name: file.name
  });
  console.log('文档上传成功:', document);
}

// 知识搜索
const searchResults = await client.search.query({
  query: '如何使用API',
  dataset_ids: [dataset.id],
  top_k: 10,
  threshold: 0.7
});

searchResults.results.forEach(result => {
  console.log(`相似度: ${result.score}`);
  console.log(`内容: ${result.content}`);
  console.log(`来源: ${result.source.document_name}`);
});
```

### 6. WebSocket实时通信

```typescript
// 创建WebSocket连接
const ws = client.createWebSocket(conversation.id);

ws.on('open', () => {
  console.log('WebSocket连接已建立');
  
  // 发送消息
  ws.send(JSON.stringify({
    type: 'message',
    content: '实时消息测试'
  }));
});

ws.on('message', (data) => {
  const message = JSON.parse(data.toString());
  console.log('收到消息:', message);
});

ws.on('error', (error) => {
  console.error('WebSocket错误:', error);
});

ws.on('close', () => {
  console.log('WebSocket连接已关闭');
});
```

## Python SDK

### 1. 基本使用

```python
import asyncio
from chatbot_sdk import ChatbotClient

async def main():
    async with ChatbotClient(api_key="your-api-key") as client:
        # 创建对话
        conversation = await client.conversations.create(
            CreateConversationRequest(title="Python SDK测试")
        )
        
        # 发送消息并处理流式响应
        async for event in client.messages.send_stream(
            conversation.id,
            SendMessageRequest(content="你好，Python SDK！")
        ):
            if event.type == "delta":
                print(event.content, end="", flush=True)
            elif event.type == "done":
                print(f"\n消息ID: {event.message_id}")

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 语音处理

```python
import asyncio
from pathlib import Path
from chatbot_sdk import ChatbotClient

async def voice_example():
    async with ChatbotClient(api_key="your-api-key") as client:
        # 语音转文字
        audio_file = Path("audio.wav")
        transcription = await client.voice.transcribe(
            audio_file=audio_file,
            language="zh-CN"
        )
        print(f"识别结果: {transcription.text}")
        print(f"置信度: {transcription.confidence}")
        
        # 文字转语音
        audio_data = await client.voice.synthesize(
            text="这是Python SDK的语音合成测试",
            voice="female",
            speed=1.0
        )
        
        # 保存音频文件
        with open("output.wav", "wb") as f:
            f.write(audio_data)

asyncio.run(voice_example())
```

### 3. 批量处理

```python
import asyncio
from chatbot_sdk import ChatbotClient, CreateConversationRequest, SendMessageRequest

async def batch_processing():
    async with ChatbotClient(api_key="your-api-key") as client:
        # 批量创建对话
        conversations = []
        for i in range(5):
            conv = await client.conversations.create(
                CreateConversationRequest(title=f"批量对话 {i+1}")
            )
            conversations.append(conv)
        
        # 并发发送消息
        async def send_message(conv_id, content):
            return await client.messages.send(
                conv_id,
                SendMessageRequest(content=content)
            )
        
        tasks = [
            send_message(conv.id, f"消息 {i+1}")
            for i, conv in enumerate(conversations)
        ]
        
        messages = await asyncio.gather(*tasks)
        
        for i, message in enumerate(messages):
            print(f"对话 {i+1} 消息ID: {message.id}")

asyncio.run(batch_processing())
```

### 4. 错误处理和重试

```python
import asyncio
from chatbot_sdk import ChatbotClient, APIError, NetworkError

async def robust_client():
    async with ChatbotClient(api_key="your-api-key") as client:
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                conversation = await client.conversations.create(
                    CreateConversationRequest(title="重试测试")
                )
                print(f"对话创建成功: {conversation.id}")
                break
                
            except APIError as e:
                if e.is_client_error:
                    print(f"客户端错误，不重试: {e.message}")
                    break
                elif e.is_rate_limit_error:
                    print("触发限流，等待后重试...")
                    await asyncio.sleep(2 ** retry_count)  # 指数退避
                else:
                    print(f"服务器错误，重试中... ({retry_count + 1}/{max_retries})")
                
            except NetworkError as e:
                print(f"网络错误，重试中... ({retry_count + 1}/{max_retries})")
                await asyncio.sleep(1)
            
            retry_count += 1
        
        if retry_count >= max_retries:
            print("达到最大重试次数，操作失败")

asyncio.run(robust_client())
```

## 常见用例

### 1. 智能客服机器人

```typescript
class CustomerServiceBot {
  private client: ChatbotClient;
  private conversations: Map<string, string> = new Map();

  constructor(apiKey: string) {
    this.client = new ChatbotClient({ apiKey });
  }

  async handleUserMessage(userId: string, message: string): Promise<string> {
    // 获取或创建用户对话
    let conversationId = this.conversations.get(userId);
    if (!conversationId) {
      const conversation = await this.client.conversations.create({
        title: `客服对话 - ${userId}`,
        metadata: { userId, type: 'customer_service' }
      });
      conversationId = conversation.id;
      this.conversations.set(userId, conversationId);
    }

    // 发送消息并获取回复
    let response = '';
    await this.client.messages.sendStream(
      conversationId,
      { content: message },
      (event) => {
        if (event.type === 'delta') {
          response += event.content || '';
        }
      }
    );

    return response;
  }
}

// 使用示例
const bot = new CustomerServiceBot('your-api-key');
const reply = await bot.handleUserMessage('user123', '我想了解产品价格');
console.log('机器人回复:', reply);
```

### 2. 文档问答系统

```python
class DocumentQA:
    def __init__(self, api_key: str):
        self.client = ChatbotClient(api_key=api_key)
        self.dataset_id = None
    
    async def setup_knowledge_base(self, documents_path: Path):
        """设置知识库"""
        # 创建数据集
        dataset = await self.client.datasets.create(
            CreateDatasetRequest(
                name="文档问答知识库",
                description="用于问答的文档集合",
                type="document"
            )
        )
        self.dataset_id = dataset.id
        
        # 批量上传文档
        for doc_path in documents_path.glob("*.pdf"):
            await self.client.datasets.upload_document(
                dataset.id,
                file_path=doc_path
            )
            print(f"已上传: {doc_path.name}")
    
    async def answer_question(self, question: str) -> str:
        """回答问题"""
        if not self.dataset_id:
            raise ValueError("请先设置知识库")
        
        # 搜索相关文档
        search_results = await self.client.search.query(
            SearchRequest(
                query=question,
                dataset_ids=[self.dataset_id],
                top_k=5,
                threshold=0.7
            )
        )
        
        # 构建上下文
        context = "\n".join([
            f"文档片段 {i+1}: {result.content}"
            for i, result in enumerate(search_results.results)
        ])
        
        # 创建临时对话
        conversation = await self.client.conversations.create(
            CreateConversationRequest(title="文档问答")
        )
        
        # 发送带上下文的问题
        prompt = f"""基于以下文档内容回答问题：

{context}

问题：{question}

请基于上述文档内容给出准确的回答。如果文档中没有相关信息，请说明无法从提供的文档中找到答案。"""
        
        message = await self.client.messages.send(
            conversation.id,
            SendMessageRequest(content=prompt)
        )
        
        return message.content

# 使用示例
async def main():
    qa_system = DocumentQA("your-api-key")
    await qa_system.setup_knowledge_base(Path("./documents"))
    
    answer = await qa_system.answer_question("产品的主要功能是什么？")
    print("回答:", answer)

asyncio.run(main())
```

### 3. 多模态聊天应用

```typescript
class MultiModalChat {
  private client: ChatbotClient;
  private conversationId: string;

  constructor(apiKey: string) {
    this.client = new ChatbotClient({ apiKey });
  }

  async initialize(): Promise<void> {
    const conversation = await this.client.conversations.create({
      title: '多模态聊天',
      metadata: { type: 'multimodal' }
    });
    this.conversationId = conversation.id;
  }

  async sendTextMessage(text: string): Promise<void> {
    await this.client.messages.sendStream(
      this.conversationId,
      { content: text, modality: 'text' },
      (event) => {
        if (event.type === 'delta') {
          this.displayMessage(event.content || '', 'assistant');
        }
      }
    );
  }

  async sendVoiceMessage(audioBlob: Blob): Promise<void> {
    // 先转换语音为文字
    const transcription = await this.client.voice.transcribe({
      audio: audioBlob,
      language: 'zh-CN'
    });

    this.displayMessage(transcription.text, 'user');

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

    this.displayMessage(text, 'assistant');
  }

  private displayMessage(content: string, role: 'user' | 'assistant'): void {
    // 在UI中显示消息
    console.log(`[${role}]: ${content}`);
  }
}
```

## 错误处理

### JavaScript/TypeScript

```typescript
import { APIError, NetworkError, ValidationError } from '@chatbot/sdk';

try {
  const conversation = await client.conversations.create({
    title: 'Test Conversation'
  });
} catch (error) {
  if (error instanceof APIError) {
    if (error.isAuthError) {
      console.log('认证失败，请重新登录');
      // 重定向到登录页面
    } else if (error.isRateLimitError) {
      console.log('请求过于频繁，请稍后再试');
      // 实现退避重试
    } else if (error.isServerError) {
      console.log('服务器错误，请稍后再试');
      // 记录错误日志
    }
  } else if (error instanceof NetworkError) {
    console.log('网络连接失败');
    // 检查网络连接
  } else if (error instanceof ValidationError) {
    console.log(`参数验证失败: ${error.message}`);
    // 显示表单验证错误
  }
}
```

### Python

```python
from chatbot_sdk import APIError, NetworkError, ValidationError

try:
    conversation = await client.conversations.create(
        CreateConversationRequest(title="Test Conversation")
    )
except APIError as e:
    if e.is_auth_error:
        print("认证失败，请重新登录")
    elif e.is_rate_limit_error:
        print("请求过于频繁，请稍后再试")
        await asyncio.sleep(5)  # 等待后重试
    elif e.is_server_error:
        print("服务器错误，请稍后再试")
except NetworkError as e:
    print(f"网络连接失败: {e.message}")
except ValidationError as e:
    print(f"参数验证失败: {e.message}")
```

## 最佳实践

### 1. 连接池和资源管理

```typescript
// JavaScript/TypeScript
class ChatbotManager {
  private static instance: ChatbotManager;
  private client: ChatbotClient;

  private constructor(apiKey: string) {
    this.client = new ChatbotClient({
      apiKey,
      timeout: 30000,
    });
  }

  static getInstance(apiKey: string): ChatbotManager {
    if (!ChatbotManager.instance) {
      ChatbotManager.instance = new ChatbotManager(apiKey);
    }
    return ChatbotManager.instance;
  }

  getClient(): ChatbotClient {
    return this.client;
  }
}
```

```python
# Python
import asyncio
from contextlib import asynccontextmanager
from chatbot_sdk import ChatbotClient

class ChatbotManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None
    
    @asynccontextmanager
    async def get_client(self):
        if not self._client:
            self._client = ChatbotClient(api_key=self.api_key)
        
        try:
            yield self._client
        finally:
            # 客户端会在上下文管理器中自动关闭
            pass

# 使用示例
manager = ChatbotManager("your-api-key")

async with manager.get_client() as client:
    conversation = await client.conversations.create(
        CreateConversationRequest(title="管理的对话")
    )
```

### 2. 请求重试和限流处理

```typescript
class RobustChatbotClient {
  private client: ChatbotClient;
  private maxRetries = 3;
  private baseDelay = 1000; // 1秒

  constructor(apiKey: string) {
    this.client = new ChatbotClient({ apiKey });
  }

  async sendMessageWithRetry(
    conversationId: string,
    content: string
  ): Promise<string> {
    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
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
          const delay = this.baseDelay * Math.pow(2, attempt);
          console.log(`触发限流，等待 ${delay}ms 后重试...`);
          await new Promise(resolve => setTimeout(resolve, delay));
          continue;
        }
        
        if (attempt === this.maxRetries - 1) {
          throw error;
        }
        
        console.log(`请求失败，重试中... (${attempt + 1}/${this.maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, this.baseDelay));
      }
    }
    
    throw new Error('达到最大重试次数');
  }
}
```

### 3. 缓存和性能优化

```python
import asyncio
from typing import Dict, Optional
from datetime import datetime, timedelta
from chatbot_sdk import ChatbotClient, Conversation

class CachedChatbotClient:
    def __init__(self, api_key: str):
        self.client = ChatbotClient(api_key=api_key)
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
    
    def clear_cache(self):
        """清理缓存"""
        self.conversation_cache.clear()
    
    def cleanup_expired_cache(self):
        """清理过期缓存"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, cached_at) in self.conversation_cache.items()
            if now - cached_at >= self.cache_ttl
        ]
        for key in expired_keys:
            del self.conversation_cache[key]
```

### 4. 监控和日志

```typescript
import { ChatbotClient, APIError } from '@chatbot/sdk';

class MonitoredChatbotClient {
  private client: ChatbotClient;
  private metrics = {
    requests: 0,
    errors: 0,
    latency: [] as number[]
  };

  constructor(apiKey: string) {
    this.client = new ChatbotClient({ apiKey });
  }

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
      
      if (error instanceof APIError) {
        console.error(`API错误 - 状态码: ${error.status}, 消息: ${error.message}`);
      } else {
        console.error(`未知错误: ${error.message}`);
      }
      
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
      averageLatency: avgLatency,
      p95Latency: this.calculatePercentile(this.metrics.latency, 0.95)
    };
  }

  private calculatePercentile(values: number[], percentile: number): number {
    if (values.length === 0) return 0;
    const sorted = values.slice().sort((a, b) => a - b);
    const index = Math.ceil(sorted.length * percentile) - 1;
    return sorted[index];
  }
}
```

这些示例展示了如何在实际项目中有效使用Chatbot SDK，包括错误处理、性能优化、缓存策略和监控等最佳实践。根据具体需求，可以进一步定制和扩展这些示例。
