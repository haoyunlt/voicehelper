# 语音优化快速实施指南

## 🚀 48小时落地清单

### Day 1 上午 (4小时) - 前端AudioWorklet重构

#### 1. 创建音频处理目录结构
```bash
mkdir -p frontend/audio/{worklets,buffers}
```

#### 2. 实现MicProcessor.js
```javascript
// frontend/audio/worklets/MicProcessor.js
class MicProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.frameSize = 320; // 20ms @ 16kHz
        this.buffer = new Float32Array(this.frameSize);
        this.bufferIndex = 0;
    }
    
    process(inputs, outputs, parameters) {
        const input = inputs[0];
        if (input.length > 0) {
            const inputChannel = input[0];
            
            for (let i = 0; i < inputChannel.length; i++) {
                this.buffer[this.bufferIndex++] = inputChannel[i];
                
                if (this.bufferIndex >= this.frameSize) {
                    // 发送完整帧到主线程
                    this.port.postMessage({
                        type: 'audioFrame',
                        data: this.buffer.slice(),
                        timestamp: currentTime
                    });
                    this.bufferIndex = 0;
                }
            }
        }
        return true;
    }
}

registerProcessor('mic-processor', MicProcessor);
```

#### 3. 实现PlayerProcessor.js
```javascript
// frontend/audio/worklets/PlayerProcessor.js
class PlayerProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.jitterBuffer = [];
        this.bufferSize = 5; // 5帧缓冲
        this.playIndex = 0;
    }
    
    process(inputs, outputs, parameters) {
        const output = outputs[0];
        
        if (this.jitterBuffer.length > this.bufferSize) {
            // 从缓冲区播放
            const frame = this.jitterBuffer.shift();
            if (frame && output.length > 0) {
                output[0].set(frame.data);
            }
        }
        
        return true;
    }
}

registerProcessor('player-processor', PlayerProcessor);
```

#### 4. 实现JitterBuffer.ts
```typescript
// frontend/audio/buffers/JitterBuffer.ts
interface AudioFrame {
    data: Float32Array;
    timestamp: number;
    sequenceNum: number;
}

export class JitterBuffer {
    private buffer: AudioFrame[] = [];
    private maxSize = 10;
    private targetSize = 5;
    
    addFrame(frame: AudioFrame) {
        // 按时间戳排序插入
        const insertIndex = this.findInsertPosition(frame.timestamp);
        this.buffer.splice(insertIndex, 0, frame);
        
        // 缓冲区溢出处理
        if (this.buffer.length > this.maxSize) {
            this.buffer.shift(); // 丢弃最旧帧
        }
    }
    
    getFrame(): AudioFrame | null {
        if (this.buffer.length >= this.targetSize) {
            return this.buffer.shift() || null;
        }
        return null;
    }
    
    private findInsertPosition(timestamp: number): number {
        for (let i = 0; i < this.buffer.length; i++) {
            if (this.buffer[i].timestamp > timestamp) {
                return i;
            }
        }
        return this.buffer.length;
    }
}
```

### Day 1 下午 (4小时) - 网关协议优化

#### 5. 统一事件封装 (Go)
```go
// backend/pkg/types/events.go
package types

import "time"

type EventEnvelope struct {
    Type  string      `json:"type"`
    Data  interface{} `json:"data"`
    Meta  EventMeta   `json:"meta"`
    Error *ErrorInfo  `json:"error,omitempty"`
}

type EventMeta struct {
    SessionID   string    `json:"session_id"`
    TraceID     string    `json:"trace_id"`
    Timestamp   int64     `json:"timestamp"`
    SequenceNum int32     `json:"sequence_num"`
}

type AudioFrame struct {
    Header AudioHeader `json:"header"`
    Data   []byte      `json:"data"`
}

type AudioHeader struct {
    SessionID   string `json:"session_id"`
    SequenceNum int32  `json:"sequence_num"`
    SampleRate  int32  `json:"sample_rate"`
    Channels    int8   `json:"channels"`
    FrameSize   int16  `json:"frame_size"`
    Timestamp   int64  `json:"timestamp"`
}
```

#### 6. WebSocket处理器优化
```go
// backend/internal/handlers/voice_ws.go
package handlers

import (
    "github.com/gorilla/websocket"
    "github.com/prometheus/client_golang/prometheus"
)

var (
    wsActiveConnections = prometheus.NewGauge(prometheus.GaugeOpts{
        Name: "ws_active_connections",
        Help: "Number of active WebSocket connections",
    })
    
    audioFramesReceived = prometheus.NewCounterVec(prometheus.CounterOpts{
        Name: "audio_frames_received_total", 
        Help: "Total audio frames received",
    }, []string{"session_id"})
)

type VoiceWSHandler struct {
    sessions map[string]*Session
    eventBus EventBus
}

func (h *VoiceWSHandler) HandleConnection(conn *websocket.Conn) {
    sessionID := generateSessionID()
    session := &Session{
        ID:   sessionID,
        Conn: conn,
        SendQueue: make(chan []byte, 100),
    }
    
    h.sessions[sessionID] = session
    wsActiveConnections.Inc()
    
    go h.handleIncoming(session)
    go h.handleOutgoing(session)
}

func (h *VoiceWSHandler) handleIncoming(session *Session) {
    defer func() {
        delete(h.sessions, session.ID)
        wsActiveConnections.Dec()
    }()
    
    for {
        _, message, err := session.Conn.ReadMessage()
        if err != nil {
            break
        }
        
        // 解析音频帧
        frame, err := parseAudioFrame(message)
        if err != nil {
            continue
        }
        
        audioFramesReceived.WithLabelValues(session.ID).Inc()
        
        // 发送到算法服务
        h.eventBus.Publish("audio_frame", frame)
    }
}
```

### Day 2 上午 (4小时) - 算法服务事件化

#### 7. LangGraph事件发射器
```python
# algo/core/events.py
import asyncio
import json
import time
from typing import Dict, Any
from enum import Enum

class EventType(Enum):
    AGENT_PLAN = "agent_plan"
    AGENT_STEP = "agent_step"
    TOOL_RESULT = "tool_result"
    SUMMARY = "summary"
    TTS_CHUNK = "tts_chunk"
    TTS_END = "tts_end"
    CANCELLED = "cancelled"

class EventEmitter:
    def __init__(self):
        self.subscribers = {}
    
    async def emit(self, event_type: EventType, data: Dict[str, Any], session_id: str):
        event = {
            "type": event_type.value,
            "data": data,
            "meta": {
                "session_id": session_id,
                "timestamp": int(time.time() * 1000),
                "trace_id": self.get_trace_id()
            }
        }
        
        # 发送到所有订阅者
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                await callback(event)
    
    def subscribe(self, event_type: EventType, callback):
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)
```

#### 8. 可中断TTS服务
```python
# algo/services/streaming_tts.py
import asyncio
from typing import AsyncIterator
import aiohttp

class StreamingTTSService:
    def __init__(self):
        self.active_sessions = set()
        self.cancellation_tokens = {}
    
    async def synthesize_streaming(self, text: str, session_id: str) -> AsyncIterator[bytes]:
        self.active_sessions.add(session_id)
        self.cancellation_tokens[session_id] = False
        
        try:
            # 分句处理
            sentences = self.split_sentences(text)
            
            for sentence in sentences:
                if self.is_cancelled(session_id):
                    await self.emit_event(EventType.CANCELLED, {}, session_id)
                    break
                
                # 调用TTS API
                async for chunk in self.call_tts_api(sentence):
                    if self.is_cancelled(session_id):
                        break
                    
                    await self.emit_event(EventType.TTS_CHUNK, {
                        "audio_data": chunk,
                        "chunk_index": self.chunk_index
                    }, session_id)
                    
                    yield chunk
            
            await self.emit_event(EventType.TTS_END, {}, session_id)
            
        finally:
            self.active_sessions.discard(session_id)
            self.cancellation_tokens.pop(session_id, None)
    
    def cancel_session(self, session_id: str):
        self.cancellation_tokens[session_id] = True
    
    def is_cancelled(self, session_id: str) -> bool:
        return self.cancellation_tokens.get(session_id, False)
    
    def split_sentences(self, text: str) -> list:
        # 简单的句子分割
        import re
        sentences = re.split(r'[.!?。！？]', text)
        return [s.strip() for s in sentences if s.strip()]
```

#### 9. VAD端点检测优化
```python
# algo/core/vad.py
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class VADConfig:
    silence_threshold: float = 0.5
    min_speech_duration: int = 300  # ms
    max_silence_duration: int = 800  # ms
    backfill_duration: int = 200     # ms
    aggressive_mode: bool = True

class EndpointDetector:
    def __init__(self, config: VADConfig):
        self.config = config
        self.state = "listening"
        self.speech_start = None
        self.silence_start = None
        self.audio_buffer = []
    
    def process_audio(self, audio_chunk: np.ndarray, timestamp: int) -> Optional[str]:
        # 计算音频能量
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        
        if energy > self.config.silence_threshold:
            # 检测到语音
            if self.state == "listening":
                self.speech_start = timestamp
                self.state = "speaking"
                return "speech_start"
            elif self.state == "silence":
                # 从静默转为语音
                self.state = "speaking"
                self.silence_start = None
        else:
            # 检测到静默
            if self.state == "speaking":
                if self.silence_start is None:
                    self.silence_start = timestamp
                elif timestamp - self.silence_start > self.config.max_silence_duration:
                    # 静默时间足够长，判定为语音结束
                    if self.speech_start and timestamp - self.speech_start > self.config.min_speech_duration:
                        self.state = "listening"
                        return "speech_end"
        
        return None
```

### Day 2 下午 (4小时) - 监控与测试

#### 10. Prometheus指标收集
```go
// backend/pkg/metrics/voice_metrics.go
package metrics

import (
    "github.com/prometheus/client_golang/prometheus"
    "github.com/prometheus/client_golang/prometheus/promauto"
)

var (
    VoiceLatencyHistogram = promauto.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "voice_latency_seconds",
            Help: "Voice processing latency by stage",
            Buckets: []float64{0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0},
        },
        []string{"stage", "session_id"},
    )
    
    AudioHealthMetrics = promauto.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "audio_health_score",
            Help: "Audio quality health metrics",
        },
        []string{"metric_type", "session_id"},
    )
    
    BargeInCounter = promauto.NewCounterVec(
        prometheus.CounterOpts{
            Name: "barge_in_total",
            Help: "Total barge-in attempts and successes",
        },
        []string{"result", "session_id"},
    )
)

// 延迟记录函数
func RecordLatency(stage string, sessionID string, duration float64) {
    VoiceLatencyHistogram.WithLabelValues(stage, sessionID).Observe(duration)
}
```

#### 11. k6压测脚本
```javascript
// tests/performance/voice_load_test.js
import ws from 'k6/ws';
import { check } from 'k6';
import { Counter, Trend } from 'k6/metrics';

const wsConnections = new Counter('ws_connections');
const e2eLatency = new Trend('e2e_latency');

export let options = {
    scenarios: {
        text_chat: {
            executor: 'constant-vus',
            vus: 70,
            duration: '5m',
            exec: 'textChatTest',
        },
        voice_chat: {
            executor: 'constant-vus',
            vus: 30, 
            duration: '5m',
            exec: 'voiceChatTest',
        }
    }
};

export function textChatTest() {
    const url = 'ws://localhost:8080/chat/stream';
    const response = ws.connect(url, {}, function (socket) {
        socket.on('open', () => {
            wsConnections.add(1);
            
            socket.send(JSON.stringify({
                type: 'text_message',
                data: { message: 'Hello, how are you?' },
                meta: { session_id: `text_${__VU}_${__ITER}` }
            }));
        });
        
        socket.on('message', (data) => {
            const message = JSON.parse(data);
            if (message.type === 'response_complete') {
                const latency = Date.now() - message.meta.start_time;
                e2eLatency.add(latency);
            }
        });
        
        socket.setTimeout(() => {
            socket.close();
        }, 30000);
    });
    
    check(response, { 'status is 101': (r) => r && r.status === 101 });
}

export function voiceChatTest() {
    const url = 'ws://localhost:8080/voice/stream';
    
    // 加载测试音频文件
    const audioData = open('./test_audio.pcm', 'b');
    
    const response = ws.connect(url, {}, function (socket) {
        socket.on('open', () => {
            wsConnections.add(1);
            
            // 分帧发送音频数据
            const frameSize = 640; // 20ms @ 16kHz
            for (let i = 0; i < audioData.length; i += frameSize) {
                const frame = audioData.slice(i, i + frameSize);
                
                // 3% 概率注入乱序
                if (Math.random() < 0.03) {
                    setTimeout(() => socket.send(frame), 50);
                } else {
                    socket.send(frame);
                }
                
                // 模拟实时发送间隔
                sleep(0.02); // 20ms
            }
        });
        
        socket.on('message', (data) => {
            // 处理TTS响应
            const message = JSON.parse(data);
            if (message.type === 'tts_end') {
                const latency = Date.now() - message.meta.start_time;
                e2eLatency.add(latency);
            }
        });
    });
    
    check(response, { 'status is 101': (r) => r && r.status === 101 });
}
```

#### 12. Playwright E2E测试
```typescript
// tests/e2e/voice_interaction.spec.ts
import { test, expect } from '@playwright/test';

test.describe('语音交互E2E测试', () => {
    test('完整语音对话流程', async ({ page }) => {
        await page.goto('/voice-chat');
        
        // 等待页面加载
        await expect(page.locator('#voice-chat-container')).toBeVisible();
        
        // 上传测试音频文件
        const audioInput = page.locator('#audio-file-input');
        await audioInput.setInputFiles('./test-data/hello.wav');
        
        // 点击开始录音按钮
        await page.click('#start-recording');
        
        // 验证录音状态
        await expect(page.locator('#recording-status')).toContainText('录音中');
        
        // 等待ASR结果
        await expect(page.locator('#asr-result')).toContainText('你好', { timeout: 5000 });
        
        // 验证LLM响应
        await expect(page.locator('#llm-response')).toBeVisible({ timeout: 10000 });
        
        // 验证TTS播放
        await expect(page.locator('#tts-player')).toHaveAttribute('playing', 'true');
        
        // 检查端到端延迟
        const latencyText = await page.locator('#e2e-latency').textContent();
        const latency = parseInt(latencyText || '0');
        expect(latency).toBeLessThan(500);
    });
    
    test('打断功能测试', async ({ page }) => {
        await page.goto('/voice-chat');
        
        // 开始对话
        await page.click('#start-conversation');
        
        // 等待TTS开始播放
        await expect(page.locator('#tts-status')).toContainText('播放中');
        
        // 模拟用户打断
        const interruptStart = Date.now();
        await page.click('#interrupt-button');
        
        // 验证打断响应
        await expect(page.locator('#tts-status')).toContainText('已取消', { timeout: 200 });
        
        const interruptLatency = Date.now() - interruptStart;
        expect(interruptLatency).toBeLessThan(120);
    });
    
    test('网络异常恢复测试', async ({ page }) => {
        await page.goto('/voice-chat');
        
        // 建立连接
        await page.click('#connect-button');
        await expect(page.locator('#connection-status')).toContainText('已连接');
        
        // 模拟网络断开
        await page.route('**/voice/stream', route => route.abort());
        
        // 验证断线检测
        await expect(page.locator('#connection-status')).toContainText('重连中', { timeout: 5000 });
        
        // 恢复网络
        await page.unroute('**/voice/stream');
        
        // 验证自动重连
        await expect(page.locator('#connection-status')).toContainText('已连接', { timeout: 10000 });
    });
});
```

## 🔧 快速验证命令

### 本地开发环境启动
```bash
# 1. 启动所有服务
make dev

# 2. 运行单元测试
make test

# 3. 运行E2E测试
make test-e2e

# 4. 运行压测
make load-test

# 5. 查看监控面板
open http://localhost:3000/grafana
```

### 性能验证脚本
```bash
#!/bin/bash
# scripts/validate_performance.sh

echo "开始性能验证..."

# 1. 检查服务健康状态
curl -f http://localhost:8080/health || exit 1

# 2. 运行延迟测试
echo "测试端到端延迟..."
LATENCY=$(curl -s http://localhost:8080/metrics | grep voice_latency | tail -1)
echo "当前延迟: $LATENCY"

# 3. 检查音频质量指标
echo "检查音频质量..."
OOO_FRAMES=$(curl -s http://localhost:8080/metrics | grep audio_ooo_frames)
DROP_FRAMES=$(curl -s http://localhost:8080/metrics | grep audio_dropped_frames)
echo "乱序帧: $OOO_FRAMES"
echo "丢失帧: $DROP_FRAMES"

# 4. 验证打断功能
echo "测试打断功能..."
BARGE_IN=$(curl -s http://localhost:8080/metrics | grep barge_in_success)
echo "打断成功率: $BARGE_IN"

echo "性能验证完成！"
```

## 📊 关键指标监控

### Grafana查询语句
```promql
# 端到端延迟P95
histogram_quantile(0.95, rate(voice_latency_seconds_bucket{stage="e2e"}[5m]))

# 五段延迟分解
rate(voice_latency_seconds_sum{stage="capture"}[5m]) / rate(voice_latency_seconds_count{stage="capture"}[5m])
rate(voice_latency_seconds_sum{stage="asr"}[5m]) / rate(voice_latency_seconds_count{stage="asr"}[5m])
rate(voice_latency_seconds_sum{stage="llm"}[5m]) / rate(voice_latency_seconds_count{stage="llm"}[5m])
rate(voice_latency_seconds_sum{stage="tts"}[5m]) / rate(voice_latency_seconds_count{stage="tts"}[5m])
rate(voice_latency_seconds_sum{stage="play"}[5m]) / rate(voice_latency_seconds_count{stage="play"}[5m])

# 音频质量健康度
rate(audio_ooo_frames_total[5m])
rate(audio_dropped_frames_total[5m])
histogram_quantile(0.95, rate(audio_jitter_seconds_bucket[5m]))

# 打断成功率
rate(barge_in_total{result="success"}[5m]) / rate(barge_in_total[5m])
```

---

## ⚡ 立即开始

1. **创建分支**: `git checkout -b v2-voice-optimization`
2. **设置环境**: `cp env.example .env.local`
3. **启动服务**: `make dev`
4. **开始第一个任务**: 实现AudioWorklet MicProcessor

让我们用48小时证明语音优化的威力！🚀
