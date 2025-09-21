
# 版本路线图（Roadmap）— 聊天机器人 & 语音助手演进

技术栈基线：**豆包 LLM（Ark） + Milvus + Python/LangChain + Go/Gin + Node.js/Next.js**  
结构：按「版本 → 目标 → 功能清单 → API/数据/基础设施变更 → SLO & 验收」展开。

---

## v1.0 — MVP（文本 Chat + RAG 首答）
**目标**：最快落地可用的企业知识问答；打通上传→检索→生成→引用的闭环。

**功能清单**
- 聊天（文本）：SSE 流式回复、会话管理（新建/重命名/归档）
- RAG 管道：文档上传、解析、切片、Embedding、写入 Milvus、TopK 检索、可选 MMR
- 引用展示：逐条引用编号与来源链接；复制代码/文本
- 基础鉴权：OIDC/JWT（单租户起步），API Key
- 观测与统计（基础版）：端到端 P95、QPS、错误率、Tokens/成本
- 安全：大小/类型白名单、敏感词拦截、基础审计日志
- 管理页（简版）：数据集列表、任务状态、用量概览

**API/数据/基础设施**
- `POST /api/chat/stream`（SSE）、`POST /api/ingest/upload`、`GET /api/search`
- Milvus：HNSW 索引（M=16, efC=200, efS=64），collection 含 `tenant_id/dataset_id/doc_id/chunk_id/vector`
- 部署：Docker Compose（本地）+ K8s（staging/prod 的 YAML 基线）
- 指标：`chat_first_token_seconds`、`retrieval_latency_seconds`、`errors_total`

**SLO & 验收**
- 纯对话首 token < 800ms；RAG 首 token < 1.2s；端到端 P95 < 2.8s
- Recall@5 ≥ 0.85（离线评测集）、引用正确率 ≥ 0.9
- 关键路径通过率 ≥ 98%；主要接口单元/集成测试覆盖 ≥ 70%

---

## v1.1 — RAG 质量与运营增强
**目标**：检索质量、可治理性、运维可视化升级。

**功能清单**
- 召回增强：多路召回（标题/全文/FAQ 混合）、轻量 **Re-ranker**（bge-reranker，可开关）
- 语义缓存：Query/Embedding 缓存（近似相似度阈值可配）
- 数据治理：文档版本与失效标记、批量重建索引、一键热/冷分区
- 评测工具：离线评测面板（Recall/nDCG/MAP）、人工抽检工作台
- 观测升级：RAG 专项图表（TopK 分数分布、空召回率、含引用回答比例）

**API/数据/基础设施**
- `POST /api/admin/reindex`、`POST /api/admin/dataset/:id/freeze`
- Milvus：按 `tags`/日期分区；索引参数在线调优接口（efSearch）
- 指标：`retrieval_topk_score`、`rag_answer_with_refs_total`、`milvus_*`

**SLO & 验收**
- 引入重排后，Top1 命中率较 v1.0 提升 ≥ 5%
- 空召回率（无可用片段）下降 ≥ 20%
- 观测仪表盘齐全，异常可定位到检索/生成环节

---

## v1.2 — 成本与性能优化（文本场景打磨）
**目标**：降低成本、提升吞吐，稳定生产用量。

**功能清单**
- 成本控制：按租户/用户配额（Tokens/天）、自动降级（长文→摘要→回答）
- Prompt 治理：系统提示/模板版本化（A/B），输出长度约束
- 并发优化：后端 Goroutine 池、限流/队列、幂等重试
- 缓存与去重：热门问答缓存（语义近似命中合并）、重复上传去重（hash）
- 可用性：断点续传（SSE 重连）、失败重试指数退避

**API/数据/基础设施**
- `POST /api/admin/quota`、`GET /api/analytics/cost`
- Redis：热点问题与最近 24h 近似缓存；对象存储清理任务
- 指标：`rate_limit_dropped_total`、`tokens_per_answer`、`cache_hit_ratio`

**SLO & 验收**
- 端到端 P95 下降 ≥ 20%；单位回答平均 Tokens 下降 ≥ 15%
- 峰值 2×QPS 稳定运行 30 分钟无错误率飙升

---

## v1.3 — 语音最小可用（Voice Phase 1）
**目标**：在文本能力上，增量支持“语音输入→文本回答→整段 TTS 播放”的闭环（无 barge-in）。

**功能清单**
- 语音通道（基础）：`WS /api/voice/stream`（start/audio/stop），流式 ASR（partial/final）
- TTS（整段）：最终文本生成后一次性合成与播放（无边合成边播）
- 字幕与卡片：ASR 字幕实时显示；引用在屏上显示
- 语音体验：端侧/服务端 VAD 参数可配；设备选择（Mic/Speaker）

**API/数据/基础设施**
- WS 协议：`asr_partial/final`、`llm_delta`（可选）、`tts_once`、`done/error`
- 新增 `modality` 字段到会话消息（text/asr/tts）
- 指标：`asr_partial_latency_seconds`、`tts_first_audio_seconds`

**SLO & 验收**
- ASR 首段 < 400ms；可感知首响（说完→开始播）< 1200ms
- 一轮语音对话成功率 ≥ 97%（无异常中断）

---

## v1.4 — 语音流式闭环 + Barge-in（Voice Phase 2）
**目标**：实现“边合成边播 + 可打断（barge-in）”，达到自然对话体验。

**功能清单**
- TTS 流式：token→句读分段→`tts_chunk` 连续播报
- **Barge-in**：用户开口或点击“停止”→ 150ms 内停止播报 + 取消 LLM
- 统一会话：文本/语音消息混排、同一个 `conversation_id` 共享上下文
- 弱网回退：TTS 故障回文本卡片、ASR 故障提示重试、自动降码率

**API/数据/基础设施**
- `POST /api/chat/cancel` + `X-Request-ID`；服务端取消转发到 LLM/TTS
- 播放端 jitter buffer（200–400ms），句读边界切片策略
- 指标：`chat_cancel_latency_seconds`、`tts_chunks_total`、`voiceFirstTts`

**SLO & 验收**
- **barge-in 生效 ≤ 150ms**；语音首响 < 700ms（稳定网）
- 端到端 P95 较 v1.3 下降 ≥ 25%；用户主观自然度提升（A/B ≥ +0.3 分）

---

## v1.5 — 语音质量/鲁棒与专业化
**目标**：提高噪声/回声环境下的稳定性，加强领域适配与多模态补充。

**功能清单**
- 噪声与回声：AEC/NS/AGC 链，在会议/车载场景启用；设备回声参考
- 热词/领域词：ASR 热词注入（联系人/SKU/地名）与租户词表管理
- 读法优化：数字/时间/货币 TN/ITN；中英夹杂 TTS 声学模型选择
- 多模态补充（可选）：图片/截图 OCR → 文档入库/检索
- 语音评测：WER/MOS 周报、抽样监听（合规前提下脱敏）

**API/数据/基础设施**
- `POST /api/admin/asr/custom-words`（租户级热词词表）
- 设备/浏览器配置页：AEC 开关、麦克风参考路径、音量自动调节
- 指标：`asr_wer_ratio`、`voice_bargein_success_rate`、`aec_active_gauge`

**SLO & 验收**
- 噪声场景下 WER 下降 ≥ 20%；barge-in 成功率 ≥ 99%
- 读数读时错误率下降 ≥ 50%（抽样）

---

## v1.6 — 工具/Agent 能力（说完要去做）
**目标**：在语音自然对话中可靠调用工具（日程、音乐、报表、IoT），形成“说–做–回”的闭环。

**功能清单**
- 工具路由：NLU/LLM 识别意图→调用专家工具→结果校验→口播总结
- 动作确认：敏感操作二次确认（口播 + 屏幕卡片）
- 可扩展工具接口：注册/权限/回显协议（审计）

**API/数据/基础设施**
- `POST /api/tools/…`（统一参数约定）；工具注册元数据表
- 审计扩充：工具调用记录、输入/输出摘要、用户确认动作
- 指标：`tool_invoke_latency_seconds`、`task_success_rate`

**SLO & 验收**
- 工具调用成功率 ≥ 98%；敏感操作误触发率 < 0.5%
- 端到端“设闹钟/查报表”等典型任务一次成功率 ≥ 95%

---

## v1.7 — 多租户与合规深水区
**目标**：支撑对外多租户、严格合规与可审核性。

**功能清单**
- 多租户：行级/集合级隔离，租户级配额/词表/策略
- 合规：音频默认不留存；抽样质检需脱敏并可关闭；审计导出/保留期
- 计费：按租户出账（ASR 分钟、TTS 字数、Tokens）

**API/数据/基础设施**
- 统一账务导出；`/api/admin/audit/export`
- 密钥管理与轮换（KMS/ExternalSecrets）；灰度与回滚流程
- 指标：租户维度成本与使用量、审计拉取次数

**SLO & 验收**
- 全链路租户隔离验收通过；合规抽检 0 问题
- 账务对账误差 < 1%

---

## v1.8 — 生态与平台化
**目标**：将能力平台化，便于集成与二开。

**功能清单**
- SDK/Embed：Web/移动端 SDK，支持快速嵌入 Chat/Voice
- 模板与预设：行业模板（客服/HR/IT/法务），提示词与评测数据集打包
- 插件/扩展商店（可选）：工具/数据源/主题扩展

**API/数据/基础设施**
- 公共 SDK（TS/Go/Python），文档与示例仓库
- 模板打包/导入导出（json/yaml）
- 指标：SDK 覆盖与错误、模板使用率

**SLO & 验收**
- SDK 集成时间 < 1 天；模板化部署 < 30 分钟
- 平台稳定性不低于核心服务（错误率差异 < 0.1%）

---

### 长期 Backlog（可穿插到迭代中）
- 检索增强：结构化表格/图像 RAG、多路融合（BM25+Dense）
- 端侧能力：轻量 ASR/TTS/关键词在边缘运行（嵌入式/离线）
- 多语言与跨语种检索（Cross-lingual）
- 安全：越权检索自动检测、内容安全对抗（Prompt 注入防护）
- 体验：情感/风格可控 TTS、说话人识别（会议侧）

---

## 排期建议（参考）
- **Q1**：v1.0（MVP）→ v1.1（RAG）  
- **Q2**：v1.2（成本性能）→ v1.3（语音 P1）  
- **Q3**：v1.4（语音 P2）→ v1.5（语音鲁棒）  
- **Q4**：v1.6（工具/Agent）→ v1.7（多租户合规）→ v1.8（平台化）

---

## 依赖与风险提示
- **供应商依赖**：ASR/TTS/LLM 限流或政策变化，需保留快速切换适配层
- **Milvus 资源**：内存/IO 压力在大规模增长时需监控分级扩容与分区策略
- **隐私合规**：麦克风/录音策略与企业/地域法规对齐，默认最小留存
- **体验边界**：语音场景必须严格控制回答长度与中断行为，避免“长篇独白”
