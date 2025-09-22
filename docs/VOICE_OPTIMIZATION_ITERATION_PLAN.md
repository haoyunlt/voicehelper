# VoiceHelper 语音链路优化迭代计划

## 🎯 迭代概述

基于业界最佳实践和性能优化需求，本次迭代专注于语音交互链路的全面优化，目标是打造低延迟、高稳定性的实时语音对话体验。

## 📅 迭代时间线

**迭代周期**: 48小时快速迭代 + 2周深度优化  
**快速交付**: 2025年9月24日  
**完整优化**: 2025年10月8日  
**当前分支**: `v2-voice-optimization`

## 🚀 核心优化目标

### 延迟优化目标
- 🎯 **端到端延迟**: < 500ms (当前 >1000ms)
- 📊 **分段延迟指标**: 
  - 采集→ASR: < 100ms
  - ASR→LLM: < 200ms  
  - LLM→TTS: < 150ms
  - TTS→播放: < 50ms
- 🔄 **打断响应**: < 120ms
- 📡 **网络抖动**: P95 < 80ms

### 稳定性目标
- 🎵 **音频质量**: 乱序/丢帧率 < 1%
- 🔗 **连接稳定**: 99.9% 可用性
- 💾 **内存占用**: 主线程占用降低 40%
- 🔊 **播放连续性**: 拼接空隙 < 20ms

## 📋 48小时快速迭代计划

### Day 1 上午 (4小时): 前端音频管线重构

#### 1.1 AudioWorklet 实现
```typescript
// 新增文件结构
frontend/
├── audio/
│   ├── worklets/
│   │   ├── MicProcessor.js      // 采集处理器
│   │   ├── PlayerProcessor.js   // 播放处理器
│   │   └── AudioWorkletNode.ts  // 封装类
│   ├── buffers/
│   │   ├── JitterBuffer.ts      // 抖动缓冲
│   │   └── RingBuffer.ts        // 环形缓冲
│   └── VoiceClient.ts           // 统一客户端
```

**实现要点**:
- [x] MicProcessor: VAD前置 + 16k/mono/PCM16 分帧(20ms)
- [x] PlayerProcessor: jitter buffer + 追帧策略
- [x] 主线程卸载: 降噪/下采样移至Worklet线程
- [x] 验证指标: Chrome任务管理器观察CPU占用下降

#### 1.2 播放侧Jitter Buffer
```typescript
interface JitterBufferConfig {
  bufferSize: number;     // 80-120ms缓冲
  maxDropFrames: number;  // 最多丢弃帧数
  adaptiveMode: boolean;  // 自适应追帧
}
```

**核心功能**:
- [x] 按时间戳排序音频帧
- [x] 小缓冲策略(80-120ms)
- [x] 追帧机制防止积压
- [x] 环形缓冲实现

### Day 1 下午 (4小时): 网关层协议优化

#### 1.3 统一Envelope格式
```go
// 统一事件封装
type EventEnvelope struct {
    Type      string                 `json:"type"`
    Data      interface{}           `json:"data"`
    Meta      EventMeta             `json:"meta"`
    Error     *ErrorInfo            `json:"error,omitempty"`
}

type EventMeta struct {
    SessionID   string    `json:"session_id"`
    TraceID     string    `json:"trace_id"`
    Timestamp   int64     `json:"timestamp"`
    SequenceNum int32     `json:"sequence_num"`
}
```

#### 1.4 WS二进制帧规范
```go
// 音频帧结构
type AudioFrame struct {
    Header AudioHeader `json:"header"`
    Data   []byte      `json:"data"`
}

type AudioHeader struct {
    SessionID    string `json:"session_id"`
    SequenceNum  int32  `json:"sequence_num"`
    SampleRate   int32  `json:"sample_rate"`
    Channels     int8   `json:"channels"`
    FrameSize    int16  `json:"frame_size"`
    Timestamp    int64  `json:"timestamp"`
}
```

#### 1.5 心跳与背压机制
```go
// 背压控制
type BackpressureController struct {
    queueSize     int32
    throttleLimit int32
    sendInterval  time.Duration
}

// 心跳检测
type HeartbeatManager struct {
    interval    time.Duration
    timeout     time.Duration
    missedBeats int32
}
```

### Day 2 上午 (4小时): 算法服务事件化

#### 2.1 LangGraph事件流
```python
# 事件类型定义
class AgentEvent:
    PLAN = "agent_plan"
    STEP = "agent_step" 
    TOOL_RESULT = "tool_result"
    SUMMARY = "summary"
    TTS_CHUNK = "tts_chunk"
    TTS_END = "tts_end"
    CANCELLED = "cancelled"

# 节点事件发射
async def emit_agent_event(event_type: str, data: dict, session_id: str):
    event = {
        "type": event_type,
        "data": data,
        "meta": {
            "session_id": session_id,
            "timestamp": int(time.time() * 1000),
            "trace_id": get_trace_id()
        }
    }
    await event_bus.publish(event)
```

#### 2.2 可中断TTS实现
```python
class StreamingTTSService:
    def __init__(self):
        self.active_sessions = {}
        self.cancellation_tokens = {}
    
    async def synthesize_streaming(self, text: str, session_id: str):
        # 分句处理
        sentences = self.split_sentences(text)
        
        for i, sentence in enumerate(sentences):
            if self.is_cancelled(session_id):
                await self.emit_event("tts_cancelled", session_id)
                break
                
            # 分片合成
            async for chunk in self.tts_client.stream_synthesize(sentence):
                if self.is_cancelled(session_id):
                    break
                await self.emit_chunk(chunk, session_id)
        
        await self.emit_event("tts_end", session_id)
```

#### 2.3 VAD端点检测优化
```python
class VADConfig:
    silence_threshold: float = 0.5    # 静默阈值
    min_speech_duration: int = 300    # 最短发声(ms)
    max_silence_duration: int = 800   # 最长静默(ms)
    backfill_duration: int = 200      # 回填时长(ms)
    aggressive_mode: bool = True      # 激进模式降低延迟

class EndpointDetector:
    def __init__(self, config: VADConfig):
        self.config = config
        self.state = "listening"
        
    def process_audio(self, audio_chunk: bytes) -> EndpointEvent:
        # VAD处理逻辑
        pass
```

### Day 2 下午 (4小时): 观测与测试

#### 2.4 五段延迟监控
```go
// 延迟指标收集
type LatencyMetrics struct {
    CaptureMs  float64 `json:"capture_ms"`
    ASRMs      float64 `json:"asr_ms"`
    LLMMs      float64 `json:"llm_ms"`
    TTSMs      float64 `json:"tts_ms"`
    PlayMs     float64 `json:"play_ms"`
    E2EMs      float64 `json:"e2e_ms"`
}

// Prometheus指标
var (
    voiceLatencyHistogram = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "voice_latency_seconds",
            Help: "Voice processing latency by stage",
            Buckets: []float64{0.05, 0.1, 0.2, 0.5, 1.0, 2.0},
        },
        []string{"stage", "session_id"},
    )
)
```

#### 2.5 音频质量监控
```go
// 音频健康度指标
type AudioHealthMetrics struct {
    OutOfOrderFrames int64   `json:"ooo_frames"`
    DroppedFrames    int64   `json:"dropped_frames"`
    JitterP95        float64 `json:"jitter_p95_ms"`
    PacketLoss       float64 `json:"packet_loss_rate"`
}
```

#### 2.6 压测脚本
```javascript
// k6压测配置
export let options = {
    scenarios: {
        text_chat: {
            executor: 'constant-vus',
            vus: 70,
            duration: '5m',
        },
        voice_chat: {
            executor: 'constant-vus', 
            vus: 30,
            duration: '5m',
        }
    }
};

// 语音压测逻辑
export function voice_test() {
    let ws = new WebSocket('ws://localhost:8080/voice/stream');
    
    // 发送固定录音文件
    let audioData = open('./test_audio.pcm', 'b');
    
    // 注入乱序/丢包
    if (Math.random() < 0.03) {
        // 3% 概率乱序
        return;
    }
    
    ws.send(audioData);
}
```

## 🔧 深度优化阶段 (2周)

### Week 1: WebRTC迁移准备

#### 3.1 传输层抽象
```typescript
// 传输接口抽象
interface AudioTransport {
    connect(config: TransportConfig): Promise<void>;
    send(data: AudioFrame): Promise<void>;
    onReceive(callback: (data: AudioFrame) => void): void;
    disconnect(): Promise<void>;
}

// WebSocket实现
class WebSocketTransport implements AudioTransport {
    // 现有WS实现
}

// WebRTC实现 (新增)
class WebRTCTransport implements AudioTransport {
    private peerConnection: RTCPeerConnection;
    private dataChannel: RTCDataChannel;
    
    async connect(config: TransportConfig) {
        // WebRTC连接建立
        this.peerConnection = new RTCPeerConnection(config.iceServers);
        this.dataChannel = this.peerConnection.createDataChannel('audio');
    }
}
```

#### 3.2 信令服务器
```go
// WebRTC信令处理
type SignalingServer struct {
    sessions map[string]*RTCSession
    mutex    sync.RWMutex
}

type RTCSession struct {
    SessionID    string
    PeerConn     *webrtc.PeerConnection
    DataChannel  *webrtc.DataChannel
    AudioTrack   *webrtc.TrackLocalStaticRTP
}

func (s *SignalingServer) HandleOffer(sessionID string, offer webrtc.SessionDescription) {
    // 处理WebRTC Offer
}
```

### Week 2: 生产级优化

#### 3.3 流式TTS集成
```python
# Deepgram TTS流式接口
class DeepgramStreamingTTS:
    def __init__(self, api_key: str):
        self.client = DeepgramClient(api_key)
        
    async def stream_synthesize(self, text: str) -> AsyncIterator[bytes]:
        async with self.client.speak.stream() as stream:
            await stream.send_text(text)
            async for chunk in stream:
                yield chunk.data

# OpenAI Realtime API集成  
class OpenAIRealtimeTTS:
    async def stream_synthesize(self, text: str) -> AsyncIterator[bytes]:
        async with websockets.connect(self.realtime_url) as ws:
            await ws.send(json.dumps({
                "type": "response.create",
                "response": {"modalities": ["audio"], "voice": "alloy"}
            }))
            
            async for message in ws:
                data = json.loads(message)
                if data["type"] == "response.audio.delta":
                    yield base64.b64decode(data["delta"])
```

#### 3.4 Grafana监控面板
```yaml
# 语音链路监控面板配置
dashboard:
  title: "VoiceHelper 语音链路监控"
  panels:
    - title: "端到端延迟漏斗"
      type: "stat"
      targets:
        - expr: "histogram_quantile(0.95, voice_latency_seconds_bucket{stage='e2e'})"
          legendFormat: "E2E P95"
        
    - title: "五段延迟分解"
      type: "graph" 
      targets:
        - expr: "voice_latency_seconds{stage='capture'}"
        - expr: "voice_latency_seconds{stage='asr'}"
        - expr: "voice_latency_seconds{stage='llm'}"
        - expr: "voice_latency_seconds{stage='tts'}"
        - expr: "voice_latency_seconds{stage='play'}"
          
    - title: "音频健康度"
      type: "singlestat"
      targets:
        - expr: "rate(audio_ooo_frames_total[5m])"
          legendFormat: "乱序率"
        - expr: "rate(audio_dropped_frames_total[5m])"
          legendFormat: "丢帧率"
          
    - title: "打断成功率"
      type: "gauge"
      targets:
        - expr: "rate(barge_in_success_total[5m]) / rate(barge_in_attempts_total[5m])"
```

#### 3.5 E2E自动化测试
```typescript
// Playwright语音E2E测试
describe('语音交互E2E测试', () => {
    test('录音-识别-回答-播放完整流程', async ({ page }) => {
        await page.goto('/voice-chat');
        
        // 上传测试录音文件
        const audioFile = './test-audio/hello.wav';
        await page.setInputFiles('#audio-input', audioFile);
        
        // 验证ASR结果
        await expect(page.locator('#asr-result')).toContainText('你好');
        
        // 验证LLM响应
        await expect(page.locator('#llm-response')).toBeVisible();
        
        // 验证TTS播放
        await expect(page.locator('#audio-player')).toHaveAttribute('playing', 'true');
        
        // 验证延迟指标
        const e2eLatency = await page.locator('#e2e-latency').textContent();
        expect(parseInt(e2eLatency)).toBeLessThan(500);
    });
    
    test('打断功能测试', async ({ page }) => {
        await page.goto('/voice-chat');
        
        // 开始TTS播放
        await page.click('#start-tts');
        await expect(page.locator('#tts-status')).toContainText('playing');
        
        // 模拟用户打断
        await page.click('#interrupt-button');
        
        // 验证打断响应时间
        const interruptLatency = await page.locator('#interrupt-latency').textContent();
        expect(parseInt(interruptLatency)).toBeLessThan(120);
        
        // 验证TTS停止
        await expect(page.locator('#tts-status')).toContainText('cancelled');
    });
    
    test('网络异常恢复测试', async ({ page }) => {
        await page.goto('/voice-chat');
        
        // 模拟网络断开
        await page.route('**/voice/stream', route => route.abort());
        
        // 验证重连机制
        await page.waitForSelector('#reconnecting-indicator');
        
        // 恢复网络
        await page.unroute('**/voice/stream');
        
        // 验证自动重连
        await expect(page.locator('#connection-status')).toContainText('connected');
    });
});
```

## 📊 成功验收标准

### 性能指标
- ✅ **端到端延迟**: P95 < 500ms
- ✅ **打断响应**: < 120ms  
- ✅ **音频质量**: 乱序/丢帧率 < 1%
- ✅ **主线程占用**: 降低 40%
- ✅ **播放连续性**: 拼接空隙 < 20ms

### 功能指标  
- ✅ **AudioWorklet**: 采集/播放管线重构完成
- ✅ **Jitter Buffer**: 抖动缓冲与追帧策略
- ✅ **事件化**: LangGraph节点事件流
- ✅ **可中断TTS**: 分片合成与取消机制
- ✅ **VAD优化**: 端点检测参数可配

### 观测指标
- ✅ **五段延迟**: 完整链路监控
- ✅ **音频健康**: 乱序/丢帧/抖动监控  
- ✅ **压测覆盖**: 文本70% + 语音30%
- ✅ **E2E测试**: 打断/重连/异常恢复
- ✅ **Grafana面板**: 实时监控看板

## 🛠️ 技术债务清理

### 代码重构
- [x] **目录收口**: 标记labs/examples，主线聚焦
- [x] **CODEOWNERS**: 代码审查责任人
- [x] **pre-commit**: 代码质量门禁
- [x] **Makefile**: 一键开发环境

### 基础设施
- [x] **docker-compose**: healthcheck + 依赖排序
- [x] **CI/CD**: Lint + 单测 + E2E + 镜像推送
- [x] **环境配置**: .env.local.example

## 🎯 里程碑检查点

### 48小时快速交付 (2025-09-24)
- ✅ AudioWorklet重构完成
- ✅ 网关协议优化
- ✅ 算法服务事件化  
- ✅ 基础监控上线

### 1周深度优化 (2025-10-01)
- ✅ WebRTC传输层抽象
- ✅ 流式TTS集成
- ✅ 完整压测覆盖
- ✅ Grafana监控面板

### 2周生产就绪 (2025-10-08)
- ✅ 性能指标全部达标
- ✅ E2E测试自动化
- ✅ 技术债务清理
- ✅ 文档完善

## 🔄 风险控制与回滚

### 风险识别
1. **AudioWorklet兼容性**: Safari/移动端支持有限
2. **WebRTC复杂性**: 信令服务器开发成本
3. **TTS流式接口**: 第三方服务依赖风险
4. **性能回归**: 优化可能引入新瓶颈

### 缓解策略
1. **渐进式迁移**: 保留WebSocket作为fallback
2. **特性开关**: 新功能可动态开关
3. **监控告警**: 关键指标阈值告警
4. **快速回滚**: 一键回滚到稳定版本

### 回滚预案
```bash
# 紧急回滚脚本
#!/bin/bash
echo "开始紧急回滚..."

# 1. 切换特性开关
curl -X POST /admin/feature-toggle -d '{"audio_worklet": false}'

# 2. 回滚代码版本
git checkout v2-stable
docker-compose up -d --force-recreate

# 3. 验证服务状态
./scripts/health_check.sh

echo "回滚完成，服务已恢复"
```

## 📚 参考资料与对标

### 业界最佳实践
- **AudioWorklet**: [MDN Web Audio API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Using_AudioWorklet)
- **WebRTC**: [Azure OpenAI Realtime](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/realtime-audio-webrtc)
- **流式TTS**: [Deepgram TTS WebSocket](https://developers.deepgram.com/docs/tts-websocket)
- **工程化参考**: [Pipecat AI Framework](https://github.com/pipecat-ai/pipecat)

### 技术选型对比
| 技术方案 | 延迟 | 稳定性 | 开发成本 | 推荐场景 |
|---------|------|--------|----------|----------|
| WebSocket | 中等 | 高 | 低 | 内网/简单场景 |
| WebRTC | 低 | 高 | 高 | 公网/实时场景 |
| SSE | 高 | 中等 | 低 | 文本流式 |
| AudioWorklet | 低 | 高 | 中等 | 音频处理 |

---

## 🎉 开始语音优化之旅！

这次迭代将显著提升VoiceHelper的语音交互体验，通过业界最佳实践的引入，我们将构建一个低延迟、高稳定性的实时语音对话系统。

让我们用48小时证明技术的力量！🚀

---

*创建时间: 2025-09-22*  
*迭代分支: v2-voice-optimization*  
*目标交付: 2025-09-24 (快速版) / 2025-10-08 (完整版)*
