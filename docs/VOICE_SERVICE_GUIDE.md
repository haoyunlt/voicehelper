# VoiceHelper AI - 语音服务使用指南

本文档介绍 VoiceHelper AI 系统中语音相关功能的使用方法和接口说明。

## 🎤 语音服务概览

VoiceHelper AI 提供完整的语音处理能力，包括：
- 语音转文字 (ASR - Automatic Speech Recognition)
- 文字转语音 (TTS - Text-to-Speech)  
- 语音对话处理
- 实时语音流处理

## 🔧 服务架构

```
用户语音输入 → 语音服务(8001) → 算法服务(8000) → AI响应 → 语音输出
```

### 服务端点
- **语音服务**: http://localhost:8001
- **算法服务**: http://localhost:8000 (处理转写后的文本)

## 📡 API 接口详解

### 1. 语音查询接口

**端点**: `POST /voice/query`  
**服务**: 语音服务 (端口 8001)

#### 请求格式
```json
{
  "conversation_id": "string",     // 会话ID (必需)
  "audio_chunk": "string",         // 音频数据 (Base64编码) (必需)
  "seq": 0,                        // 音频序列号 (必需)
  "codec": "opus",                 // 音频编码格式 (可选，默认opus)
  "sample_rate": 16000             // 采样率 (可选，默认16000)
}
```

#### 音频格式要求
- **编码格式**: opus, mp3, wav, flac
- **采样率**: 16000Hz (推荐), 8000Hz, 44100Hz
- **声道**: 单声道 (mono)
- **数据格式**: Base64 编码的音频数据

#### 示例请求
```bash
curl -X POST http://localhost:8001/voice/query \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "conv_001",
    "audio_chunk": "UklGRnoGAABXQVZFZm10IBAAAAABAAEA...",
    "seq": 1,
    "codec": "wav",
    "sample_rate": 16000
  }'
```

### 2. 文本查询接口 (处理语音转写结果)

**端点**: `POST /query`  
**服务**: 算法服务 (端口 8000)

#### 请求格式
```json
{
  "messages": [
    {
      "role": "user",
      "content": "语音转写的文本内容"
    }
  ],
  "temperature": 0.3,
  "max_tokens": 1024
}
```

#### 示例：处理语音转写结果
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user", 
        "content": "你好，请帮我查询今天的天气情况"
      }
    ]
  }'
```

## 🎯 语音功能测试

### 测试场景 1: 模拟语音转写处理

当您有语音转写结果时，可以直接通过算法服务处理：

```bash
# 模拟语音转写结果: "这是模拟的语音转写结果"
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {
        "role": "user",
        "content": "这是模拟的语音转写结果，请帮我分析一下这句话的含义"
      }
    ]
  }'
```

**响应示例**:
```json
{"type": "content", "content": "我理解您提到了语音转写结果...", "refs": null}
{"type": "end", "content": null, "refs": null}
```

### 测试场景 2: 健康检查

```bash
# 检查语音服务状态
curl http://localhost:8001/health

# 检查算法服务状态  
curl http://localhost:8000/health
```

## 🔊 语音处理流程

### 完整语音对话流程

1. **音频采集** → 客户端录制音频
2. **音频编码** → 转换为支持的格式 (opus/wav/mp3)
3. **Base64编码** → 将音频数据编码为字符串
4. **发送请求** → 调用 `/voice/query` 接口
5. **语音转文字** → 服务端 ASR 处理
6. **文本理解** → 算法服务处理转写文本
7. **生成回复** → AI 生成响应内容
8. **文字转语音** → TTS 生成音频回复
9. **返回结果** → 客户端播放音频

### 数据流示例

```
[用户语音] 
    ↓ (录音)
[音频文件: audio.wav]
    ↓ (Base64编码)
[编码字符串: "UklGRnoGAAB..."]
    ↓ (API调用)
[语音服务: /voice/query]
    ↓ (ASR转写)
[文本: "你好，今天天气怎么样？"]
    ↓ (文本处理)
[算法服务: /query]
    ↓ (AI理解+生成)
[回复: "今天天气晴朗，温度适宜..."]
    ↓ (TTS合成)
[音频回复]
```

## 🛠️ 开发集成示例

### JavaScript/TypeScript 示例

```typescript
// 语音录制和发送
class VoiceService {
  private conversationId: string;
  private sequenceNumber: number = 0;

  async sendVoiceQuery(audioBlob: Blob): Promise<any> {
    // 转换音频为Base64
    const audioBase64 = await this.blobToBase64(audioBlob);
    
    const request = {
      conversation_id: this.conversationId,
      audio_chunk: audioBase64,
      seq: ++this.sequenceNumber,
      codec: "wav",
      sample_rate: 16000
    };

    const response = await fetch('http://localhost:8001/voice/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request)
    });

    return response.json();
  }

  private async blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = (reader.result as string).split(',')[1];
        resolve(base64);
      };
      reader.readAsDataURL(blob);
    });
  }
}
```

### Python 示例

```python
import base64
import requests
import json

class VoiceClient:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.conversation_id = "conv_001"
        self.sequence = 0

    def send_voice_query(self, audio_file_path: str):
        # 读取音频文件并编码
        with open(audio_file_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        
        self.sequence += 1
        payload = {
            "conversation_id": self.conversation_id,
            "audio_chunk": audio_data,
            "seq": self.sequence,
            "codec": "wav",
            "sample_rate": 16000
        }
        
        response = requests.post(
            f"{self.base_url}/voice/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        return response.json()

    def send_text_query(self, text: str):
        """直接发送文本到算法服务"""
        payload = {
            "messages": [
                {"role": "user", "content": text}
            ]
        }
        
        response = requests.post(
            "http://localhost:8000/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        return response.text

# 使用示例
client = VoiceClient()

# 发送语音文件
# result = client.send_voice_query("test_audio.wav")

# 或直接发送文本（模拟语音转写结果）
result = client.send_text_query("这是模拟的语音转写结果")
print(result)
```

## 🔍 错误处理

### 常见错误码

| 错误码 | 描述 | 解决方案 |
|--------|------|----------|
| 411001 | 音频格式无效 | 检查音频编码和Base64格式 |
| 411002 | 音频数据为空 | 确保提供有效的音频数据 |
| 411003 | 不支持的编码格式 | 使用支持的格式: opus, wav, mp3, flac |
| 411004 | 采样率不支持 | 使用支持的采样率: 8000, 16000, 44100 |

### 错误响应示例

```json
{
  "code": 411001,
  "message": "Invalid Audio Format",
  "description": "音频格式无效",
  "http_status": 400,
  "category": "Voice",
  "service": "Voice",
  "custom_message": "没有提供音频数据"
}
```

## 🎛️ 配置说明

### 环境变量配置

在 `.env` 文件中配置语音相关参数：

```bash
# 语音服务配置
VOICE_PORT=8001
AZURE_SPEECH_KEY=your-azure-speech-key
AZURE_SPEECH_REGION=eastus

# 音频处理配置
DEFAULT_SAMPLE_RATE=16000
DEFAULT_CODEC=opus
MAX_AUDIO_DURATION=60  # 秒
```

### 支持的音频格式

| 格式 | 编码 | 推荐用途 |
|------|------|----------|
| opus | opus | 实时语音 (低延迟) |
| wav | pcm | 高质量录音 |
| mp3 | mp3 | 压缩音频 |
| flac | flac | 无损音频 |

## 🚀 性能优化建议

1. **音频质量**
   - 使用 16kHz 采样率获得最佳识别效果
   - 单声道录音减少数据量
   - 控制音频时长在 60 秒以内

2. **网络优化**
   - 使用 opus 编码减少传输数据
   - 实现音频分块传输
   - 添加重试机制处理网络异常

3. **用户体验**
   - 实现实时语音识别反馈
   - 添加录音状态指示器
   - 提供语音输入的文本预览

## 📝 注意事项

1. **隐私保护**: 语音数据包含敏感信息，确保传输加密
2. **存储策略**: 考虑音频数据的存储和清理策略
3. **并发处理**: 语音处理可能较耗时，注意并发控制
4. **错误恢复**: 实现音频传输失败的重试机制

---

*最后更新时间: 2025-09-22*  
*文档版本: v1.9.0*
