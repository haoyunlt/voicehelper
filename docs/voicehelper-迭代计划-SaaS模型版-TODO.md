# voicehelper 迭代计划（SaaS 模型调用版 / 无电话接入）
> 版本：2025-09-22  · 本文件可直接作为 **Cursor** 项目的 `TODO.md` / 需求文档使用  
> 目标：**全部基于第三方托管模型/API**（LLM/ASR/TTS/多模态/重排/安全等），**不考虑电话/SIP**；最快速把“网页/桌面”实时语音助手做实并可验证。

---

## 目录
- [0. 总览与范围](#0-总览与范围)
- [1. 能力与外部依赖（全部第三方托管）](#1-能力与外部依赖全部第三方托管)
- [2. 里程碑与时间盒](#2-里程碑与时间盒)
- [3. 版本详细计划（可复制为任务）](#3-版本详细计划可复制为任务)
  - [v0.1 实时语音 MVP（Web/桌面）](#v01-实时语音-mvpweb桌面)
  - [v0.2 Agent + RAG（只读工具闭环）](#v02-agent--rag只读工具闭环)
  - [v0.3 MCP 工具与策略安全](#v03-mcp-工具与策略安全)
  - [v0.4 多模态（看屏幕/看世界）](#v04-多模态看屏幕看世界)
  - [v0.5 SRE/成本/DX](#v05-sre成本dx)
- [4. 验收指标（Definition of Done）](#4-验收指标definition-of-done)
- [5. 项目结构与分支规划](#5-项目结构与分支规划)
- [6. 环境变量与密钥（env.example）](#6-环境变量与密钥envexample)
- [7. PR 模板 / 代码规范 / 自动化](#7-pr-模板--代码规范--自动化)
- [8. 风险与回退策略](#8-风险与回退策略)

---

## 0. 总览与范围
- **渠道**：Web（浏览器，支持 WeChat 内置浏览器兼容清单）/ 桌面（Electron 可选）。
- **排除项**：不做电话/SIP/PSTN 接入；不做本地自训模型与自建 ASR/TTS 推理。
- **首要目标**：在 **第三方 Realtime/ASR/TTS API** 上实现“语音⇄语音”实时对话，含 **barge-in（打断）**、低延迟、可观测与回放评测。
- **核心约束**：所有智能能力均来自托管服务（可切 Provider），我们负责**编排、状态、工具安全、观测与体验**。

---

## 1. 能力与外部依赖（全部第三方托管）
> 选型采用“可拔插 ProviderAdapter”，**至少 2 家可用**以便回退；默认首选在 `.env` 指定。

- **Realtime 双工（语音⇄语音）**：
  - Provider A：OpenAI Realtime（WebRTC/WS）
  - Provider B：Deepgram Realtime / ElevenLabs Realtime（回退）
- **ASR（转写）**：OpenAI（/audio.transcriptions）、Deepgram、AssemblyAI（回退）
- **TTS（合成）**：OpenAI TTS、ElevenLabs、Azure Speech（回退）
- **LLM（文本推理）**：OpenAI（GPT-4o 级），回退：Anthropic/Google 兼容 SDK（仅文本/图像）
- **Embeddings**：OpenAI Embeddings（回退：VoyageAI）
- **重排/排序**：Cohere Rerank 或 OpenAI 文本判别式模型
- **安全/内容审核**：OpenAI Moderation（回退：Azure Content Safety）
- **向量/知识库（托管）**：Qdrant Cloud / Pinecone / Zilliz Cloud（三选一，可互换）
- **监控/日志**：Grafana Cloud（或自托 Prom+Grafana）、Sentry（前端与服务端），OpenTelemetry SDK 做埋点

> 注意：本文档不会实现供应商具体调用细节，但**每一项任务**都要求封装在 `ProviderAdapter` 下并含回退逻辑。

---

## 2. 里程碑与时间盒
| 版本 | 时间盒（建议） | 主题 | 关键交付 |
|---|---:|---|---|
| **v0.1** | 2 周 | 实时语音 MVP | 语音⇄语音、打断、观测、回放与指标曲线 |
| **v0.2** | 2–3 周 | Agent + RAG | 检索增强与“只读工具”闭环，引用可解释 |
| **v0.3** | 2–3 周 | MCP + 策略安全 | 5 个 MCP 工具、目的约束、人审/二次确认 |
| **v0.4** | 3 周 | 多模态 | 屏幕/相机低帧理解与三场景 Demo |
| **v0.5** | 2 周 | SRE/成本/DX | SLA/预算/降级、SDK/CLI/模板仓 |

---

## 3. 版本详细计划（可复制为任务）

### v0.1 实时语音 MVP（Web/桌面）
**范围**：基于第三方 Realtime/ASR/TTS 完成“语音⇄语音”对话，含 barge-in、回放与观测。

#### 需求
- [ ] 前端采集音频（AudioWorklet），显示输入/输出电平
- [ ] Realtime 双工连接（ProviderAdapter）：WebRTC 优先，WS 备选
- [ ] **barge-in（打断）**：本地 VAD 触发立即停播；服务端终止 TTS 流
- [ ] STT/LLM/TTS 全链路打点：`mic_in → asr_first_char → llm_first_token → tts_first_byte → speaker_out`
- [ ] 对话回放器：录制 event/timestamp，支持 *.jsonl 回放复测
- [ ] 观测面板：Grafana（P50/P95、错误率、打断响应时延）
- [ ] 错误码与降级：主 Provider 超时 → 自动切回退 Provider

#### 交付物
- [ ] `apps/web` 最小 UI（开始/停止、打断提示、引用块）
- [ ] `packages/providers/*`：Realtime/ASR/TTS 适配器与回退
- [ ] `apps/evals/voice-replay`：回放 CLI + 指标统计脚本
- [ ] Grafana 仪表盘 JSON 导出

#### 验收
- [ ] 安静环境 **E2E 延迟 P50 < 600ms、P95 < 1s**
- [ ] **打断到停声 P95 < 200ms**
- [ ] 150 条脚本（闲聊/问答/打断/纠错）回放报告

---

### v0.2 Agent + RAG（只读工具闭环）
**范围**：不改变语音链路，在 LLM 侧引入 RAG 与只读类工具（检索/查询），全部调用托管 API。

#### 需求
- [ ] RAG 管线：文档清洗→切片→向量化（Embeddings）→入库（Qdrant/Pinecone/Zilliz）
- [ ] 多路召回（向量+关键字）+ 重排（Cohere/OpenAI 判别式）
- [ ] 答复必须含 **引用来源**；证据不足 → **不可回答/追问**
- [ ] Agent 控制流：单循环 ReAct + 明确预算（最大轮次/最大 token）
- [ ] 评测：RR@k、nDCG、引用命中率、不可回答触发率

#### 交付物
- [ ] `services/rag`：索引与查询 API（走第三方托管向量库）
- [ ] `apps/evals/rag`：离线评测脚本与报告
- [ ] RAG 数据治理指南（去重、更新、撤稿流程）

#### 验收
- [ ] 任务成功率 ≥ 75%（基于黄金集）
- [ ] **引用命中率 ≥ 90%**；不可回答策略命中 ≥ 95%

---

### v0.3 MCP 工具与策略安全
**范围**：把“外部系统能力”经 **MCP** 暴露给 Agent 使用；仅允许低风险/只读类工具；引入目的约束与人审。

#### 需求
- [ ] MCP 客户端：允许连接远程 MCP Server（HTTP/STDIO）
- [ ] 上线 5 个高价值工具（只读）：日历查询、检索 API、数据库查询（只读）、两项第三方 API
- [ ] **目的约束**（purpose binding）：LLM 在工具调用前需生成“目的声明”，与工具白名单匹配才放行
- [ ] 高风险路径（若未来开放写操作）需 **TTS 复述 + 用户确认**
- [ ] 每个工具 20 个用例、错误注入（超时/参数非法/回退）

#### 交付物
- [ ] `apps/agent`：控制流/策略模块（含目的约束拦截器）
- [ ] `apps/evals/tools`：工具回放与通过率报告
- [ ] 审计日志：`tool_call_id/trace_id/args/redactions`

#### 验收
- [ ] 工具调用通过率 ≥ 90%，异常可回退
- [ ] 审计日志与告警规则覆盖关键路径

---

### v0.4 多模态（看屏幕/看世界）
**范围**：Web 端**屏幕共享/相机**低帧采样（1–2 FPS）+ 托管多模态模型（图像理解/屏幕理解），实现 3 个稳定场景。

#### 需求
- [ ] 屏幕共享/相机帧采集与脱敏（遮盖敏感区域）
- [ ] 多模态提问：画面解释、步骤指导、对象识别；输出带引用与指引
- [ ] 流式 UI：先给“粗要点”，随后补充细节
- [ ] 评测：首帧理解时延、任务成功率、用户反馈评分

#### 交付物
- [ ] `apps/web`：多模态输入组件与权限提示
- [ ] `apps/evals/mm`：三场景脚本与报告（装机指导/网页比价/物体说明）

#### 验收
- [ ] **首帧理解 P95 < 1.5s**
- [ ] 三场景稳定通过回放

---

### v0.5 SRE/成本/DX
**范围**：可观测、SLA、成本与降级；开发者体验（SDK/CLI/模板仓）。

#### 需求
- [ ] OTel SDK 全链路埋点 → Grafana Cloud（或自托）
- [ ] 成本看板：每会话成本、各 Provider 成本占比、缓存命中率
- [ ] 降级策略：配额/速率/预算触发时，自动切小模型或只读模式
- [ ] JS/Python SDK：流式/重连/打断/引用/回放接口
- [ ] CLI：快速录音、回放、采样、采集日志
- [ ] 模板仓：最小可跑的 Web + Agent + RAG 工程

#### 交付物
- [ ] `packages/sdk-js` / `packages/sdk-py` / `apps/cli`
- [ ] `templates/minimal-realtime`（接入 ProviderAdapter 的最小示例）
- [ ] SLA/预算/降级策略文档与演练脚本

#### 验收
- [ ] **SLA ≥ 99.5%**，重大回归可用回放 100% 复现
- [ ] 10 分钟新手上手跑通 Demo

---

## 4. 验收指标（Definition of Done）
- **实时链路**：E2E 延迟（P50/P95），**打断到停声**（P95），错误率，重连成功率
- **RAG**：RR@k、nDCG、引用命中率、不可回答触发率
- **工具**：工具调用通过率、异常回退率、审计覆盖率
- **多模态**：首帧理解时延、任务成功率、用户满意度评分
- **SRE/成本**：SLA、每会话成本、缓存命中、降级触发次数

---

## 5. 项目结构与分支规划
```
voicehelper/
├─ apps/
│  ├─ web/                 # 实时语音 + 多模态前端
│  ├─ agent/               # 控制流、策略与目的约束
│  ├─ evals/               # 回放/离线评测（voice-replay, rag, tools, mm）
│  └─ gateway/             # Realtime/WS 反向代理与统一鉴权
├─ services/
│  ├─ rag/                 # 托管向量库适配层（Qdrant/Pinecone/Zilliz）
│  └─ audit/               # 审计与脱敏（如需）
├─ packages/
│  ├─ providers/           # ProviderAdapter（Realtime/ASR/TTS/LLM/Emb/Rerank/Safety）
│  ├─ sdk-js/              # JS SDK
│  └─ sdk-py/              # Python SDK
├─ templates/              # 最小可跑模板
├─ ops/
│  ├─ grafana-dashboards/  # 仪表盘 JSON
│  └─ otel/                # OTel 配置
└─ docs/
   ├─ ADRs/                # 架构决策记录
   └─ playbooks/           # 故障/演练脚本
```
**分支策略**
- `main`：受保护，仅 squash-merge
- `release/x.y.z`：发版分支
- `feat/*`、`fix/*`、`chore/*`：常规特性/修复/杂项
- 标签：`area:[web|agent|providers|rag|evals]`, `prio:[p0|p1|p2]`, `risk:[low|med|high]`

---

## 6. 环境变量与密钥（env.example）
```
# 主/备 Provider
PROVIDER_REALTIME=OpenAIRealtime          # 备选：DeepgramRealtime / ElevenLabsRealtime
PROVIDER_ASR=OpenAITranscribe             # 备选：Deepgram / AssemblyAI
PROVIDER_TTS=OpenAITTS                    # 备选：ElevenLabs / AzureSpeech
PROVIDER_LLM=OpenAIChat                   # 备选：Anthropic / Google
PROVIDER_EMBEDDINGS=OpenAIEmbeddings      # 备选：VoyageAI
PROVIDER_RERANK=CohereRerank              # 备选：OpenAIClassifier
PROVIDER_SAFETY=OpenAIModeration          # 备选：AzureContentSafety

# API Keys（按需）
OPENAI_API_KEY=...
DEEPGRAM_API_KEY=...
ASSEMBLYAI_API_KEY=...
ELEVENLABS_API_KEY=...
AZURE_SPEECH_KEY=...

# 向量库（选其一）
QDRANT_API_URL=...       QDRANT_API_KEY=...
PINECONE_API_URL=...     PINECONE_API_KEY=...
ZILLIZ_API_URL=...       ZILLIZ_API_KEY=...

# 观测/日志
OTEL_EXPORTER_OTLP_ENDPOINT=...
GRAFANA_CLOUD_API_KEY=...
SENTRY_DSN=...
```

---

## 7. PR 模板 / 代码规范 / 自动化
**PR 模板**（摘要）
- 背景 & 需求链接
- 变更点（前后对比截图/录屏）
- 指标影响（延迟/错误/成本）
- 风险与回退
- 测试用例/回放集

**规范**
- 提交信息：`<type>(<area>): <subject>  #[issue]`
- Types：feat/fix/refactor/chore/docs/test/build/ci
- Lint：ESLint + Prettier（前端/Node），ruff/black（Python）
- 安全扫描：`npm audit` / `pip-audit` / 依赖许可检查

**自动化（建议）**
- CI：安装依赖 → Lint → 单测 → 关键脚本演示（无头录屏） → 产出回放/指标 artifact
- CD：仅模板与静态前端走自动部署；服务需人工审批

---

## 8. 风险与回退策略
- **Provider 不稳定/限流** → 立即**切回退 Provider**，记录降级事件；若全不可用，降级为“仅文本问答/仅检索”
- **ASR 噪声/回声** → 开启浏览器 AEC/NS/AGC；必要时降低帧长、启用更强的端点检测策略
- **成本失控** → 预算告警 + 模型路由切小模型；关闭多模态/重排等高成本分支
- **数据/合规** → 对上传帧/截图做脱敏；保留期与访问控制；审计日志全量落盘

> **提示**：如需我把本计划拆成 GitHub Issues + Projects 看板，请直接在仓库开一个空白 Project，并授予我维护权限即可。
