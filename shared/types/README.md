# VoiceHelper Types

VoiceHelper 项目的统一类型定义包，提供了完整的 TypeScript 类型定义，确保前端、后端、SDK 之间的类型一致性。

## 安装

```bash
npm install @voicehelper/types
```

## 使用

### 基础用法

```typescript
import { BaseResponse, User, Message, VoiceSession } from '@voicehelper/types';

// API 响应类型
const response: BaseResponse<User> = {
  success: true,
  data: {
    user_id: 'usr_123',
    username: 'john_doe',
    created_at: '2023-01-01T00:00:00Z',
    status: 'active'
  },
  timestamp: '2023-01-01T00:00:00Z'
};

// 消息类型
const message: Message = {
  message_id: 'msg_123',
  conversation_id: 'conv_123',
  role: 'user',
  content: 'Hello, world!',
  content_type: 'text',
  created_at: '2023-01-01T00:00:00Z'
};
```

### 事件类型

```typescript
import { ChatEvent, VoiceEvent, SystemEvent } from '@voicehelper/types';

// 聊天事件
const chatEvent: ChatMessageStartEvent = {
  event_id: 'evt_123',
  event_type: 'chat.message.start',
  timestamp: '2023-01-01T00:00:00Z',
  source: 'voicehelper',
  user_id: 'usr_123',
  conversation_id: 'conv_123',
  data: {
    message_id: 'msg_123',
    role: 'assistant',
    model: 'gpt-4'
  }
};

// 语音事件
const voiceEvent: VoiceSessionStartEvent = {
  event_id: 'evt_456',
  event_type: 'voice.session.start',
  timestamp: '2023-01-01T00:00:00Z',
  source: 'voicehelper',
  user_id: 'usr_123',
  session_id: 'sess_123',
  data: {
    session_id: 'sess_123',
    settings: {
      language: 'en-US',
      voice_id: 'voice_1',
      sample_rate: 16000,
      channels: 1,
      format: 'pcm',
      vad_enabled: true,
      noise_suppression: true,
      echo_cancellation: true
    },
    capabilities: {
      supported_languages: ['en-US', 'zh-CN'],
      supported_voices: [],
      supported_formats: [],
      features: []
    }
  }
};
```

### 类型检查工具

```typescript
import { isBaseResponse, isUser, isMessage, toBaseResponse } from '@voicehelper/types';

// 类型检查
if (isBaseResponse(data)) {
  console.log('Valid response:', data.success);
}

if (isUser(userData)) {
  console.log('User:', userData.username);
}

// 创建标准响应
const response = toBaseResponse({ id: 1, name: 'test' }, true, 'Success');

// 创建错误响应
const errorResponse = toErrorResponse(
  { code: 'NOT_FOUND', message: 'User not found' },
  'User lookup failed'
);
```

### 枚举类型

```typescript
import { 
  MessageRole, 
  MessageContentType, 
  VoiceSessionStatus, 
  ConversationStatus 
} from '@voicehelper/types';

// 使用枚举
const message: Message = {
  message_id: 'msg_123',
  conversation_id: 'conv_123',
  role: MessageRole.USER,
  content: 'Hello',
  content_type: MessageContentType.TEXT,
  created_at: '2023-01-01T00:00:00Z'
};

const session: VoiceSession = {
  session_id: 'sess_123',
  user_id: 'usr_123',
  status: VoiceSessionStatus.ACTIVE,
  created_at: '2023-01-01T00:00:00Z',
  settings: {
    language: 'en-US',
    voice_id: 'voice_1',
    sample_rate: 16000,
    channels: 1,
    format: 'pcm',
    vad_enabled: true,
    noise_suppression: true,
    echo_cancellation: true
  }
};
```

## 类型分类

### API 类型 (`api.d.ts`)
- 基础响应类型：`BaseResponse`, `ErrorInfo`, `PaginatedResponse`
- 用户相关：`User`, `UserPreferences`, `VoiceSettings`
- 会话相关：`Conversation`, `Message`, `MessageMetadata`
- 语音相关：`VoiceSession`, `AudioChunk`, `TranscriptionResult`
- 工具调用：`ToolCall`, `ToolResult`, `ToolDefinition`
- 数据集相关：`Dataset`, `Document`, `DocumentChunk`
- 系统监控：`SystemMetrics`, `HealthCheck`

### 事件类型 (`events.d.ts`)
- 基础事件：`BaseEvent`, `EventPayload`
- 连接事件：`ConnectionEvent`, `ConnectionOpenedEvent`, `ConnectionClosedEvent`
- 聊天事件：`ChatEvent`, `ChatMessageStartEvent`, `ChatMessageChunkEvent`
- 语音事件：`VoiceEvent`, `VoiceSessionStartEvent`, `VoiceAudioChunkEvent`
- 工具事件：`ToolEvent`, `ToolCallStartEvent`, `ToolCallEndEvent`
- 系统事件：`SystemEvent`, `SystemHealthCheckEvent`, `SystemAlertEvent`
- 用户事件：`UserEvent`, `UserLoginEvent`, `UserActivityEvent`
- 数据事件：`DataEvent`, `DocumentUploadEvent`, `SearchQueryEvent`

### 通用类型 (`common.d.ts`)
- 工具类型：`Nullable`, `Optional`, `DeepPartial`, `DeepRequired`
- 时间类型：`Timestamp`, `Duration`, `TimeRange`, `DateRange`
- 文件类型：`FileInfo`, `FileUpload`, `FileReference`
- 网络类型：`NetworkInfo`, `RequestContext`, `ClientInfo`
- 配置类型：`ConfigValue`, `ValidationRule`, `AppConfig`
- 状态类型：`State`, `AsyncState`, `CacheState`
- 权限类型：`Permission`, `Role`, `AccessControl`
- 队列类型：`Task`, `Queue`, `QueueSettings`
- 缓存类型：`CacheEntry`, `CacheStats`, `CacheConfig`
- 度量类型：`Metric`, `Counter`, `Gauge`, `Histogram`

## 版本兼容性

当前版本：`1.0.0`

支持的版本：
- `1.0.0` - 当前版本

## 开发指南

### 添加新类型

1. 在相应的 `.d.ts` 文件中添加类型定义
2. 在 `index.d.ts` 中导出新类型
3. 添加类型检查函数（如需要）
4. 更新 README.md 文档
5. 更新版本号

### 类型命名规范

- 接口名使用 PascalCase：`BaseResponse`, `UserInfo`
- 类型别名使用 PascalCase：`ApiResponse`, `EventType`
- 枚举使用 PascalCase：`MessageRole`, `VoiceSessionStatus`
- 属性名使用 snake_case：`user_id`, `created_at`
- 事件类型使用点分隔：`chat.message.start`, `voice.session.end`

### 向后兼容性

- 不要删除现有的类型定义
- 不要修改现有属性的类型（除非是 bug 修复）
- 新增属性应该是可选的
- 使用 `@deprecated` 标记废弃的类型

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request 来改进类型定义。

## 更新日志

### 1.0.0 (2023-09-23)
- 初始版本
- 完整的 API 类型定义
- 事件系统类型定义
- 通用工具类型定义
- 类型检查和转换工具函数
