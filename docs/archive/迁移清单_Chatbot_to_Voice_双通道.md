# 从 Chatbot 到 Voice 的迁移清单（支持文本与语音双通道）

> 适配现有架构：豆包 LLM（Ark）+ Milvus + Python/LangChain + Go/Gin + Node.js/Next.js  
> 目标：在保留文本聊天的同时新增 **语音输入/输出**、**可打断（barge-in）**、**低首响**，统一会话。

---

## 0. 目标与边界
- **目标**：文本与语音并存、同一会话可自由切换；端到端首响 < 700ms（稳定网）。
- **新增模块**：VAD、ASR、TTS、Barge-in、（可选）AEC/NS/AGC。
- **不改动**：RAG（Milvus）、对话编排（Go→Python）、租户/鉴权/审计。

---

## 1. 架构增量（Delta）
```
Browser/App (Mic/Speaker)
  ├─ WebAudio/WebRTC/WS 流
  ├─ Wake Word(可选, 端侧)
  └─ 双通道UI(Text + Voice)
       ↓
Go/Gin API Gateway
  ├─ /api/voice/stream  (WebSocket)
  ├─ /api/tts/stream    (可选独立)
  └─ /api/asr/stream    (可选独立)
       ↓
Python Algo (FastAPI)
  ├─ VAD/ASR/TTS 适配器(厂商SDK/HTTP)
  ├─ RAG (Milvus) / Prompt 语音化
  └─ Streaming Token ↔ TTS Chunks
       ↓
LLM: 豆包(Ark)   Milvus: 向量检索
```

---

## 2. 模块与接口
### 2.1 前端（Next.js）
- 麦克风权限/设备选择、Opus 16k 单声道推流、字幕展示。
- **Barge-in**：播放 TTS 时若检测到用户开口，立即暂停播放器并调用 `/api/chat/cancel`。
- 文本/语音输入无缝切换，统一 `conversation_id`。

### 2.2 后端（Go/Gin）——新增与变更
- **WS `/api/voice/stream`**：承载音频上行、ASR 增量转写、LLM 文本 delta、TTS 音频分片。
- 取消协议：`POST /api/chat/cancel` + `X-Request-ID`（停止 Ark 与 TTS）。
- 速率限制：音频分钟 + LLM Token，按用户/租户配额。

**WS 协议建议**
```jsonc
// Client -> Server
{"type":"start","codec":"opus","sample_rate":16000,"conversation_id":"c_123"}
{"type":"audio","seq":1,"chunk":"<base64-opus>"}
...
{"type":"stop"}

// Server -> Client
{"type":"asr_partial","seq":1,"text":"..."}        // 实时字幕
{"type":"asr_final","seq":1,"text":"..."}          // 稳定分段
{"type":"llm_delta","text":"..."}                  // 可选：文本先行
{"type":"tts_chunk","seq":n,"pcm":"<base64-pcm>"}  // 边合成边播
{"type":"refs","items":[{chunk_id,source,score}]}  // RAG 引用
{"type":"done"}
```

### 2.3 算法（Python/FastAPI）
- 流式 ASR：partial→final；支持热词注入（人名/SKU/地名）。
- 流式 TTS：token→句读分段→分片合成，拼播无爆音。
- Prompt 语音友好：短句口语、先结论后步骤≤3、引用只在屏幕展示。

---

## 3. 延迟预算（目标）
| 环节 | 目标 |
|---|---|
| VAD 结束判定 | < 300ms |
| ASR 首段 | < 400ms |
| LLM 首 token（文本） | < 800ms（RAG 时 < 1.2s） |
| TTS 首音 | < 300–500ms |
| 可感知首响（说完→开始播） | < 700ms |

---

## 4. 监控与 KPI
- 语音链：WER、ASR RTF、VAD 延迟、TTS 首音/中断时延、误/漏唤醒率（如有唤醒）。
- 对话链：首 token、端到端 P95、RAG Recall@5、幻觉率。
- 稳定性：barge-in 成功率、取消请求 ≤150ms 生效率 ≥99%。
- 成本：ASR 分钟、TTS 秒数、LLM Token 按租户统计。

---

## 5. 数据与合规
- 会话消息：新增 `modality: "text"|"audio"|"asr"|"tts"`；默认不长期存音频。
- 热词词表：租户级维护；调用 ASR 时动态携带。
- 用户提示：麦克风指示、录音说明、随时可关闭的显著入口。

---

## 6. 迁移步骤（分阶段）
**Phase 1：最小可用（2 周）**
- 前端推流（Opus 16k），Algo 对接流式 ASR；final 文本→RAG→豆包→文本→整段 TTS；基础指标采集。

**Phase 2：流式闭环（2–3 周）**
- LLM token→TTS chunk 边播；实现 barge-in；VAD 参数可配；统一会话混排。

**Phase 3：质量与鲁棒（2–3 周）**
- ASR 热词/领域增强、TTS 停顿优化、弱网降级（文本/降码率）、隐私合规完善、E2E 评测。

---

## 7. 测试要点（增量）
- 功能：语音→文本→RAG→文本→TTS 全链路；引用编号与屏幕一致。
- 中断：用户说“停/取消”，TTS ≤150ms 停止，Ark 请求被取消。
- 异常：ASR/TTS/LLM/Milvus/网络异常的回退路径与用户话术。
- 性能：首响 <700ms；P95 达标；CPU/内存无泄漏。
- 可用性：耳机/外放切换、权限弹窗、降级体验。

---

## 8. 语音化 Prompt 模板（示例）
```
系统：你是语音助手。请用简短口语回答，先给结论，再给最多3步建议。如答案基于资料，请在屏幕上显示来源编号，但口头只说“依据已显示在屏幕上”。找不到依据就直说，并给出下一步建议。
用户：{query}
检索片段（用于屏幕展示）：{context}
输出：口语短句 + 可中断
```

---

## 9. 验收标准（DoD）
- 文本与语音双通道在同一会话内无缝切换。
- 端到端首响 <700ms；barge-in ≤150ms 生效。
- RAG 引用可视化与口播一致，无越权检索。
- 异常与回退路径覆盖齐全；Trace 贯穿；语音专项指标达标。
- 隐私合规提示清晰，音频留存最小化。