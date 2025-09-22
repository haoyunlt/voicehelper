
# voicehelper 逐文件改造清单与代码级建议（对标最新语音助手方案）
**版本**：v0.1 • **日期**：2025-09-22 • **适用仓库**：[`haoyunlt/voicehelper`](https://github.com/haoyunlt/voicehelper)

> 目的：在不颠覆现有目录的前提下，补齐“实时语音（WebRTC+语义VAD+可中断TTS）/可观测/评测/多提供商路由/MCP 工具链”等第一梯队能力，并给出**逐目录逐文件**的落地清单与**代码级建议**。

---

## 0. 仓库快照（用于定位）
已存在的关键目录与文件（来自仓库根目录可见内容）：
- `.github/workflows/`（CI）
- `admin/`、`algo/`、`backend/`、`common/`、`deploy/`
- `desktop/`、`developer-portal/`、`docs/`、`frontend/`、`mobile/`
- `browser-extension/src/content/`、`sdks/`、`scripts/`、`tests/`
- 根文件：`Makefile`、`README.md`、`docker-compose.local.yml`、`env.example`、`QUICK_MODEL_SETUP.md` 等

> 注：若与本地分支不一致，请以**本地实际结构**为准；以下方案均采用“**MOD**=修改、**ADD**=新增、**DEL**=删除/迁移”标记。

---

## 1) backend/（Python，建议 FastAPI/Starlette 架构）

### 1.1 目录重构（ADD）
```
backend/app/
  ├─ main.py                          # 应用入口（FastAPI + 路由挂载 + 中间件）
  ├─ config.py                        # 配置中心（pydantic-settings）
  ├─ deps.py                          # 依赖注入（会话、路由策略、限流）
  ├─ routers/
  │   ├─ health.py                    # 健康检查/就绪检查
  │   ├─ chat.py                      # 文本对话（SSE）
  │   ├─ realtime.py                  # 语音会话控制（WebRTC/WS 信令）
  │   ├─ tools_mcp.py                 # MCP 工具链暴露/调试
  │   └─ admin.py                     # 管理面接口（回放/指标/AB 切换）
  ├─ services/
  │   ├─ stt/
  │   │   ├─ base.py                  # 统一接口（start_stream/ingest/finish）
  │   │   ├─ deepgram.py              # 云端 STT 实现（默认）
  │   │   └─ riva.py                  # 本地化 STT（可选）
  │   ├─ tts/
  │   │   ├─ base.py                  # 统一接口（stream/cancel）
  │   │   ├─ aura.py                  # Deepgram Aura-2（或 ElevenLabs）
  │   │   └─ openai_rt.py             # OpenAI Realtime 语音输出
  │   ├─ vad/
  │   │   ├─ semantic_vad.py          # 语义端点检测（可接 Daily/自研）
  │   ├─ llm/
  │   │   ├─ router.py                # 按延迟/复杂度/成本路由模型
  │   │   ├─ openai_realtime.py       # Realtime/Responses API 适配
  │   │   └─ claude.py                # 复杂推理/备用路径
  │   ├─ webrtc/
  │   │   ├─ signaling.py             # 房间/令牌/会话状态（对接 LiveKit/Daily/自建 SFU）
  │   │   └─ mixers.py                # 音频混音（系统播报/提示音/回声消除钩子）
  │   ├─ memory/
  │   │   ├─ short_term.py            # 短期对话记忆（会话级）
  │   │   └─ long_term.py             # 长期/情节记忆（向量+KV）
  │   ├─ tools_mcp/
  │   │   ├─ registry.py              # 工具注册/鉴权/配额
  │   │   └─ impls/…                  # 工单/日程/内部 API 等具体工具
  │   ├─ obs/
  │   │   ├─ tracing.py               # OpenTelemetry + 分段时延采集
  │   │   ├─ metrics.py               # Prometheus 指标
  │   │   └─ audit.py                 # 会话审计/录音脱敏（PII）
  │   └─ utils/
  │       ├─ audio.py                 # PCM/Opus 编解码、分片
  │       └─ errors.py                # 业务错误码/可恢复异常
  ├─ models/
  │   ├─ schemas.py                   # Pydantic DTO（Turn/Segment/ToolCall）
  │   └─ events.py                    # 事件枚举（START/END/BARGE/ERROR…）
  └─ tests/                           # 单测/回放/合成噪声集
```

### 1.2 关键文件建议

- **backend/app/main.py（MOD/ADD）**
  - 启用中间件：GZipMiddleware、CORSMiddleware、TrustedHostMiddleware。
  - 接入 OpenTelemetry（otel SDK + otlp exporter）与统一日志格式（JSON）。
  - 统一异常处理为结构化错误：{{"code","message","trace_id"}}。

- **backend/app/routers/realtime.py（ADD）**
  - 职责：会话创建/销毁、令牌签发、打断控制。
  - 提供端点：
    - POST /realtime/session：创建会话，返回 WebRTC/WS 的 ephemeral token 与 STT/TTS/LLM 路由策略。
    - POST /realtime/cancel-tts：可中断 TTS；实现为发布 CANCEL_TTS(session_id) 到内部总线。
    - POST /realtime/barge-in：手动触发 barge-in（用于调试/极端场景）。

  示例骨架：
  ```py
  # backend/app/routers/realtime.py
  from fastapi import APIRouter, Depends
  from app.services.webrtc.signaling import create_session_token
  from app.services.obs.tracing import span_ctx
  from app.models.schemas import SessionCreateReq, SessionCreateResp

  r = APIRouter(prefix="/realtime", tags=["realtime"])

  @r.post("/session", response_model=SessionCreateResp)
  async def create_session(req: SessionCreateReq):
      with span_ctx("session.create"):
          token, policy = await create_session_token(req.user_id, req.client_caps)
          return SessionCreateResp(token=token, policy=policy)
  ```

- **backend/app/services/stt/base.py（ADD）**
  - 统一接口，支持流式分片输入（16kHz PCM/Opus）。
  - 要求：首字时间 ≤200ms（P95），错误可重试（网络/配额）。

  ```py
  class STTBase:
      async def start_stream(self, lang: str, hints: list[str] | None = None): ...
      async def ingest(self, pcm_chunk: bytes, ts_ms: int): ...
      async def finish(self) -> "Transcription": ...
  ```

- **backend/app/services/tts/base.py（ADD）**
  - 需提供 stream(text) -> AsyncIterator[audio_chunk] 与 cancel(session_id)。
  - 实现细节：将 TTS 输出拆分为小片段（≤200ms）下发，便于 barge-in 立停。

- **backend/app/services/vad/semantic_vad.py（ADD）**
  - 引入语义端点检测（可对接 Daily Smart Turn 或自研 LSTM/Transformer VAD），阈值与动态噪声门限可配置。
  - 输出：SPEECH_START/END 事件 + 端点时间戳，联动 CANCEL_TTS 与说话人切换。
  ```py
  class SemanticVAD:
      def __init__(self, min_speech_ms=120, min_silence_ms=200, energy_thresh=-45):
          ...

      def feed(self, pcm_chunk: bytes, ts_ms: int) -> list[tuple[str,int]]:
          # 返回事件列表，如： [("SPEECH_START", ts), ("SPEECH_END", ts)]
          # SPEECH_START 触发 -> 发布 CANCEL_TTS(session_id)
          return []
  ```

- **backend/app/services/llm/router.py（ADD）**
  - 路由策略：
    - latency_budget_ms 与 complexity_score（基于 ASR 结果长度/实体数/语义困难度）。
    - 简单任务 → 小模型/缓存；复杂任务 → 高阶模型（Claude/OpenAI）。
  - 支持在线 AB 与熔断降级（配额/区域受限时）。

- **backend/app/services/webrtc/signaling.py（ADD）**
  - 若采用 LiveKit/Daily：这里仅签发房间/参与者 token，不自建 SFU。
  - 若走 OpenAI Realtime（WebRTC 直连）：此处创建 ephemeral session 并下发给前端。

- **backend/app/services/obs/metrics.py（ADD）**
  - 指标：
    - 分段时延（capture_ms, stt_ms, llm_ms, tts_ms, e2e_ms）
    - Barge-in 成功率、端点误判率（早/迟 ≥200ms）
    - ASR WER/CER（按口音/噪声分桶）

- **backend/app/tests/**（ADD/MOD）
  - test_replay_audio.py：将 1k 条真实/合成音频做回放评测；产出基线报表（CSV/JSON）。
  - test_cancel_barge.py：验证 用户发声→100ms 内 TTS 停止。

---

## 2) frontend/（建议 Next.js/React）

### 2.1 目录建议（ADD）
```
frontend/src/
  ├─ app/(voice)/page.tsx            # 语音会话页面
  ├─ components/VoiceConsole.tsx     # 调试控制台（指标/事件流）
  ├─ hooks/useWebRTC.ts              # 建连/轨道/重连/弱网自适应
  ├─ hooks/useBargeIn.ts             # 监听本地麦克风能量 + 端点事件
  ├─ lib/audio-worklet.ts            # AudioWorklet：回声消除/AEC 钩子
  ├─ lib/metrics.ts                  # 前端指标上报（TTFB/音频首包）
  ├─ store/voice.ts                  # 会话状态
  └─ styles/voice.css
```

### 2.2 关键文件建议
- **hooks/useWebRTC.ts（ADD）**
  - createSession()：调用后端 /realtime/session 获取 token + policy。
  - 使用 WebRTC（LiveKit/Daily/OpenAI Realtime）建立双工音频流；开启 onTrack 时即时播放 TTS 分片。
  - 弱网：启用 maxBitrate/degradationPreference、自动降采样。

  ```ts
  export function useWebRTC() {
    const [pc, setPc] = useState<RTCPeerConnection | null>(null);
    const start = async () => {
      const { token } = await fetch('/api/realtime/session').then(r => r.json());
      // 省略：根据后端返回选择 LiveKit/Daily/OpenAI Realtime 的 SDK 建连
    };
    const stop = async () => pc?.close();
    return { start, stop };
  }
  ```

- **hooks/useBargeIn.ts（ADD）**
  - 监听本地语音能量/端点事件 → 触发 POST /realtime/cancel-tts。
  - 对误触发设置最短说话时长阈值与去抖逻辑。

- **components/VoiceConsole.tsx（ADD）**
  - 实时展示：端到端时延、STT 文本、路由命中、barge-in 次数、丢包率等。

---

## 3) sdks/

- **sdks/js/**（ADD/MOD）
  - 导出 VoiceClient：封装 /realtime/session、WebRTC 建连、事件流、cancelTts()。
  - TypeScript 类型对齐 backend/app/models/schemas.py。

- **sdks/python/**（ADD）
  - VoiceClient（Python）：便于服务端对接/批量回放。

---

## 4) browser-extension/

- **src/content/**（MOD）
  - 新增页面拾音/播报开关、快捷键（PTT：push-to-talk）。
  - 注入脚本捕获页面上下文（URL/选中文本）→ 作为工具输入。

---

## 5) mobile/ 与 desktop/

- **mobile/**：集成 WebRTC SDK，保持与前端一致的会话/打断语义；iOS 注意 AVAudioSession 类别与回声消除。
- **desktop/**（Electron）：main 进程桥接音频权限；渲染进程复用前端 hooks。

---

## 6) deploy/

- **docker-compose.local.yml（MOD）**
  - 增加 prometheus, grafana, otel-collector、（可选）livekit（若本地自建）。
  - ulimits 与 sysctl：提高文件描述符与 UDP buffer。
- **K8s/Helm（ADD）**
  - charts/voicehelper/：多副本部署、水平扩缩、就绪探针、podAntiAffinity。

---

## 7) .github/workflows/

- **ci.yml（MOD/ADD）**
  - python -m pytest + 覆盖率阈值（≥70%）
  - 前端 typecheck + eslint；构建预览产物（vercel preview 可选）
  - locust/k6 压测作业（小流量）；失败阈值：P95 e2e < 700ms

---

## 8) tests/

- **tests/replay/**（ADD）：1k 条不同噪声/口音/语速音频；含 指令/中断 混合场景。
- **tests/e2e/voice_latency.test.ts（ADD）**：端到端时延断言；录屏/录音保存到 artifact。
- **tests/tooling/**（ADD）：MCP 工具 AB 回归集。

---

## 9) 代码级关键点（实现要点与片段）

### 9.1 可中断 TTS（后端）
```py
# backend/app/services/tts/base.py
from typing import AsyncIterator

class TTSBase:
    async def stream(self, session_id: str, text: str) -> AsyncIterator[bytes]: ...
    async def cancel(self, session_id: str) -> None: ...

# 典型实现：
# 1) TTS 在内部维护 session_id -> asyncio.Event(cancelled)
# 2) 生成音频时每 100~200ms 产出一片；发送前检查 event.is_set()，立刻 return 结束。
```

### 9.2 语义 VAD + barge-in（后端）
```py
# backend/app/services/vad/semantic_vad.py
class SemanticVAD:
    def __init__(self, min_speech_ms=120, min_silence_ms=200, energy_thresh=-45):
        ...

    def feed(self, pcm_chunk: bytes, ts_ms: int) -> list[tuple[str,int]]:
        # 返回事件列表，如： [("SPEECH_START", ts), ("SPEECH_END", ts)]
        # SPEECH_START 触发 -> 发布 CANCEL_TTS(session_id)
        return []
```

### 9.3 LLM 路由（后端）
```py
# backend/app/services/llm/router.py
def route(request) -> str:
    score = estimate_complexity(request.text)
    if score < 0.4 and request.latency_budget_ms < 600:
        return "openai:gpt-mini"     # 低成本/快
    if score < 0.7:
        return "openai:gpt-4o-mini"  # 中等
    return "claude-3.7"              # 高推理/较慢
```

### 9.4 前端：本地能量阈值触发打断
```ts
// frontend/src/hooks/useBargeIn.ts
export function useBargeIn({ sessionId }: { sessionId: string }) {
  useEffect(() => {
    let running = true;
    (async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = new AudioContext();
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      src.connect(analyser);
      const buf = new Uint8Array(analyser.fftSize);
      while (running) {
        analyser.getByteTimeDomainData(buf);
        const rms = Math.sqrt(buf.reduce((s, v) => s + (v - 128) ** 2, 0) / buf.length);
        if (rms > 10 /* 简化阈值 */) {
          fetch('/api/realtime/cancel-tts', { method: 'POST', body: JSON.stringify({ sessionId }) });
        }
        await new Promise(r => setTimeout(r, 50));
      }
    })();
    return () => { running = false; };
  }, [sessionId]);
}
```

---

## 10) 验收指标（写入看板/PR 门槛）
- 端到端时延：P95 < 700ms；首包语音 < 300ms。
- barge-in：用户发声后 ≤ 100ms 停止 TTS（成功率 ≥ 98%）。
- ASR：通用集 WER 下降 ≥ 5pp（相对基线）。
- 稳定性：丢包 3%、抖动 50ms 条件下对话可用；重连 < 2s。
- 成本：语义缓存命中率 ≥ 60%，单位会话成本 -30%。

---

## 11) 与业界方案对齐点（落地映射）
- WebRTC 优先：浏览器侧用 WebRTC 打造“超低时延+抖动缓冲”；服务间可保留 WebSocket。
- 语义 VAD：采用 Daily/LiveKit 的语义端点或自研门限 + LLM 辅助。
- STT/TTS 双栈：云端（Deepgram/OpenAI） + 本地化（Riva）；按地域/延迟/成本动态路由。
- OpenAI Realtime / MCP：前端直连语音/多模态；后端统一签发短期令牌并提供工具生态。

> 以上对齐点可与项目 README 中语音/WebSocket/GraphRAG/MCP 等描述交叉验证，并对应本方案的实现位点。

---

## 12) 迁移与兼容
- 保留原有 WebSocket 语音通道作为 后备链路；新客默认走 WebRTC。
- 配置层允许“按租户/按会话”切换 STT/TTS/LLM 提供商。
- 渐进迁移策略：灰度 5% → 25% → 50% → 100%，每个阶段对照看板指标。

---

## 13) 清单式任务分解（可直接贴到项目管理）
- [ ] backend/app/main.py：接入 Otel + 统一日志 + CORS/压缩中间件
- [ ] backend/app/routers/realtime.py：会话/打断端点
- [ ] backend/app/services/stt/{{base,deepgram,riva}}.py
- [ ] backend/app/services/tts/{{base,aura,openai_rt}}.py（实现分片 + cancel）
- [ ] backend/app/services/vad/semantic_vad.py（SPEECH_START→CANCEL_TTS）
- [ ] backend/app/services/llm/{{router,openai_realtime,claude}}.py
- [ ] backend/app/services/webrtc/signaling.py（LiveKit/Daily/OpenAI RT）
- [ ] backend/app/services/obs/{{tracing,metrics,audit}}.py
- [ ] frontend/src/hooks/{{useWebRTC,useBargeIn}}.ts + lib/audio-worklet.ts
- [ ] sdks/js：VoiceClient 封装 + 类型
- [ ] tests：回放评测 + e2e 端到端时延 + 工具回归
- [ ] deploy：Prometheus/Grafana/Otel-Collector +（可选）LiveKit 服务
- [ ] .github/workflows：CI/压测阈值门槛
- [ ] docs：开发者快速上手（令牌/会话/事件流/错误码）

---

## 14) 配置样例（env）
```env
# 语音/多媒体
VOICE_MODE=webrtc            # webrtc | websocket
STT_PROVIDER=deepgram        # deepgram | riva
TTS_PROVIDER=aura            # aura | openai_rt | elevenlabs
LLM_ROUTER=default           # default | lowcost-first | latency-first
LATENCY_BUDGET_P95=700

# 路由/密钥
OPENAI_API_KEY=...
DEEPGRAM_API_KEY=...
RIVA_SERVER=...
LIVEKIT_KEY=...
LIVEKIT_SECRET=...
```

---

### 附录 A：错误码建议
- VOICE-4001：端点检测超时
- VOICE-4002：打断失败（超时/重复）
- VOICE-5001：STT 提供商故障（可重试）
- VOICE-5002：TTS 取消通道异常
- VOICE-5003：LLM 路由降级触发

### 附录 B：日志字段规范
- trace_id / session_id / turn_id / seq
- phase：capture/stt/llm/tts/e2e
- dur_ms、pkt_loss、jitter_ms、barge_in（bool）
- provider、model、region、cost_token

---

> 如需逐文件到函数级别的改造（逐行 diff/重构建议），请将 backend/ 和 frontend/ 的关键文件（音频链路/路由/会话管理）文件名发我，我会在本清单基础上补齐到代码片段与接口签名逐处点评。
