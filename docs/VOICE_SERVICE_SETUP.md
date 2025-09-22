# 语音服务设置指南

## 概述

本项目已实现完整的语音服务功能，支持多个ASR（语音识别）和TTS（语音合成）提供商，包括：

- **ASR提供商**: OpenAI Whisper, Azure Speech, 本地speech_recognition
- **TTS提供商**: OpenAI TTS, Azure Speech, Edge TTS（免费）
- **功能特性**: 多提供商降级、缓存、VAD语音检测、流式处理

## 🚀 快速开始

### 1. 安装依赖

```bash
# 在项目根目录
cd algo
pip install -r requirements.txt
```

如果遇到SSL问题，可以使用conda或者配置pip信任源：

```bash
# 使用conda（推荐）
conda install speech-recognition edge-tts webrtcvad pydub

# 或使用pip信任源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple speech-recognition edge-tts webrtcvad pydub
```

### 2. 配置API密钥

在 `env.unified` 文件中配置以下密钥（至少配置一个）：

```bash
# OpenAI (推荐用于高质量ASR)
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1

# Azure Speech (推荐用于生产环境，成本低)
AZURE_SPEECH_KEY=your-azure-speech-key-here
AZURE_SPEECH_REGION=eastus

# 免费选项（无需配置）
# - Edge TTS: 自动可用
# - 本地ASR: 使用Google Web Speech API
```

### 3. 测试语音服务

```bash
cd algo
python tests/test_voice_integration.py
```

### 4. 运行示例

```bash
cd algo
python examples/voice_service_example.py
```

## 📋 提供商对比

| 提供商 | ASR | TTS | 成本 | 质量 | 特点 |
|--------|-----|-----|------|------|------|
| OpenAI | ✅ | ✅ | 中等 | 高 | Whisper模型，多语言支持 |
| Azure Speech | ✅ | ✅ | 低 | 高 | 企业级，稳定性好 |
| Edge TTS | ❌ | ✅ | 免费 | 好 | 微软免费TTS，中文效果好 |
| 本地ASR | ✅ | ❌ | 免费 | 中等 | Google Web Speech，有限制 |

## 🔧 配置选项

### 环境变量配置

```bash
# 基础配置
ENABLE_VOICE_PROCESSING=true
VOICE_DEFAULT_LANGUAGE=zh-CN
VOICE_DEFAULT_VOICE=zh-CN-XiaoxiaoNeural

# 性能配置
VOICE_ASR_TIMEOUT=10.0
VOICE_TTS_TIMEOUT=15.0
VOICE_ENABLE_VAD=true          # 语音活动检测
VOICE_ENABLE_CACHE=true        # TTS缓存
VOICE_CACHE_TTL=3600          # 缓存时间（秒）

# 音频配置
VOICE_SAMPLE_RATE=16000
VOICE_CHANNELS=1
VOICE_SAMPLE_WIDTH=2

# 提供商特定配置
OPENAI_ASR_MODEL=whisper-1
OPENAI_TTS_MODEL=tts-1
OPENAI_VOICE=alloy

AZURE_VOICE=zh-CN-XiaoxiaoNeural
AZURE_SPEECH_REGION=eastus

EDGE_TTS_VOICE=zh-CN-XiaoxiaoNeural
```

## 💡 使用示例

### ASR使用

```python
from core.voice_config import create_voice_config
from core.enhanced_voice_services import EnhancedASRService

# 创建ASR服务
voice_config = create_voice_config()
asr_service = EnhancedASRService(voice_config)

# 转写音频
with open('audio.wav', 'rb') as f:
    audio_data = f.read()

text = await asr_service.transcribe(audio_data, language='zh-CN')
print(f'识别结果: {text}')
```

### TTS使用

```python
from core.voice_config import create_voice_config
from core.enhanced_voice_services import EnhancedTTSService

# 创建TTS服务
voice_config = create_voice_config()
tts_service = EnhancedTTSService(voice_config)

# 合成语音
text = "你好，欢迎使用语音助手！"
audio_data = await tts_service.synthesize(text)

# 保存音频
with open('output.mp3', 'wb') as f:
    f.write(audio_data)
```

### 流式TTS

```python
# 流式合成（低延迟）
async for chunk in tts_service.synthesize_streaming(text):
    if chunk:
        # 实时播放音频块
        play_audio_chunk(chunk)
```

### 集成语音服务

```python
from core.voice import VoiceService

# 创建语音服务（包含ASR+TTS+RAG集成）
voice_service = VoiceService(retrieve_service)

# 处理语音查询
async for response in voice_service.process_voice_query(request):
    if response.type == "asr_partial":
        print(f"部分识别: {response.text}")
    elif response.type == "asr_final":
        print(f"最终识别: {response.text}")
    elif response.type == "tts_chunk":
        # 播放语音块
        play_audio_chunk(base64.b64decode(response.pcm))
```

## 🛠️ 架构说明

### 核心组件

1. **VoiceProviderFactory**: 提供商工厂，创建不同的ASR/TTS提供商
2. **EnhancedASRService**: 增强ASR服务，支持多提供商降级
3. **EnhancedTTSService**: 增强TTS服务，支持缓存和降级
4. **EnhancedVoiceService**: 集成语音服务，整合ASR+TTS+RAG
5. **VoiceConfig**: 语音配置管理

### 提供商实现

- **OpenAIASRProvider**: OpenAI Whisper ASR
- **AzureASRProvider**: Azure Speech ASR
- **LocalASRProvider**: 本地speech_recognition ASR
- **OpenAITTSProvider**: OpenAI TTS
- **AzureTTSProvider**: Azure Speech TTS
- **EdgeTTSProvider**: Edge TTS（免费）

### 特性

- **多提供商降级**: 主要提供商失败时自动切换到备用提供商
- **缓存机制**: TTS结果缓存，避免重复合成
- **VAD检测**: 语音活动检测，过滤静音
- **流式处理**: 支持流式ASR和TTS，降低延迟
- **统计监控**: 提供详细的使用统计和性能指标

## 🔍 故障排除

### 常见问题

1. **SSL证书错误**
   ```bash
   # 使用信任源安装
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple speech-recognition
   ```

2. **依赖包缺失**
   ```bash
   # 检查requirements.txt中的依赖
   pip install -r requirements.txt
   ```

3. **API密钥无效**
   ```bash
   # 检查环境变量
   echo $OPENAI_API_KEY
   echo $AZURE_SPEECH_KEY
   ```

4. **音频格式问题**
   - 确保音频是16kHz, 16-bit, 单声道WAV格式
   - 使用pydub进行格式转换

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看详细日志
voice_service = EnhancedVoiceService(config)
```

### 性能优化

1. **启用缓存**: `VOICE_ENABLE_CACHE=true`
2. **调整超时**: 根据网络情况调整 `VOICE_ASR_TIMEOUT` 和 `VOICE_TTS_TIMEOUT`
3. **选择合适的提供商**: Azure成本低，OpenAI质量高，Edge TTS免费
4. **使用VAD**: `VOICE_ENABLE_VAD=true` 过滤静音，提高效率

## 📊 监控和统计

```python
# 获取服务统计
stats = voice_service.get_stats()
print(f"ASR统计: {stats['asr_stats']}")
print(f"TTS统计: {stats['tts_stats']}")

# 检查提供商状态
from core.voice_config import get_voice_provider_status
status = get_voice_provider_status()
print(f"提供商状态: {status}")
```

## 🚀 生产部署建议

1. **配置多个提供商**: 确保有主要和备用提供商
2. **监控API配额**: 设置使用量告警
3. **启用缓存**: 减少API调用和成本
4. **调整超时**: 根据网络环境优化
5. **日志监控**: 监控错误率和延迟
6. **定期清理**: 清理过期缓存和会话

## 📝 更新日志

- **v1.0.0**: 初始版本，支持基本ASR/TTS功能
- **v1.1.0**: 添加多提供商支持和降级机制
- **v1.2.0**: 添加缓存和VAD功能
- **v1.3.0**: 添加流式处理和性能优化
- **v1.4.0**: 集成RAG和完整语音对话功能

## 🤝 贡献

欢迎提交Issue和Pull Request来改进语音服务功能。

## 📄 许可证

本项目遵循MIT许可证。
