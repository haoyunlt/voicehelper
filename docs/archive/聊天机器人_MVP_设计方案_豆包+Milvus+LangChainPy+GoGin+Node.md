# 一、概览
- **模型**：豆包大模型（火山方舟 Ark，OpenAI-Compatible 接口）
- **向量数据库**：Milvus（支持 HNSW / IVF_FLAT / HNSW_PQ 等索引）
- **算法/编排**：Python，基于 LangChain（`langchain`, `langchain-milvus`）
- **后端**：Go，Gin 框架（REST + SSE / WS），负责鉴权、路由、限流、观测与编排调度到算法服务
- **前端**：Node.js（Next.js/SSR + Edge Runtime 可选），内嵌 Chat UI 与简版管理后台
- **部署**：Kubernetes（或 Docker Compose 起步），分环境（dev/staging/prod），以 API Key + OIDC 统一鉴权
- **MVP 能力**：对话、RAG（上传→解析→切片→向量化→检索→生成）、会话记忆（短期）、流式输出、基础运营统计与成本可观测

---

# 二、总体架构
```
[Browser/SDK]
  └─ Next.js Chat UI (SSE/WS, 上传/检索, 登录)
       └─ BFF(可选)+API 调用 → Go/Gin Backend (Auth/Rate Limit/Orchestration)
            ├─ Python Algo Service（LangChain Pipeline，HTTP/gRPC）
            │    ├─ Ingest：解析→切片→Embedding→Milvus
            │    ├─ Query：检索→重排(可选)→提示组装→LLM(豆包)
            │    └─ Memory：短期会话、查询缓存
            ├─ Milvus（向量相似检索）
            ├─ 豆包 LLM（Ark ChatCompletions）
            ├─ Redis / Postgres（会话、用户、KV 缓存）
            └─ Obs：OTel/Prometheus/Grafana + 日志（ELK/Vector）
```

---

# 三、模块划分

## 3.1 前端（Node.js / Next.js）
**职责**：会话与消息 UI、上传入口、引用片段展示、反馈收集、登录鉴权、基础统计面板

**关键点**：
- **页面**：
  - `/chat`：主对话界面（消息流、引用折叠/展开、复制、反馈👍/👎）
  - `/datasets`：数据源/文档列表与索引状态（仅管理员）
  - `/analytics`：会话数、平均响应时长、Token/成本（简版）
- **交互**：SSE 优先（降级 WS），打字机流式渲染；文件/URL 上传（多文件队列）
- **鉴权**：OIDC/企业 SSO，可选 JWT Cookie；前端路由守卫 + SSR 边缘缓存（用户/租户隔离）
- **可用性**：失败重试（指数退避）、断线自动恢复、输入节流（防抖 300ms）
- **可观测**：Web Vitals、前端埋点（会话开始/结束、反馈、错误码、耗时）

**与后端接口（示例）**：
- `POST /api/chat/stream`（SSE）：{conversation_id?, messages[], tools?}
- `POST /api/ingest/upload`：multipart 文件上传，返回 task_id
- `GET  /api/ingest/tasks/:id`：查看解析/切片/入库进度
- `GET  /api/search?q=...`：仅检索预览（开发/验收用）

---

## 3.2 后端（Go / Gin）
**职责**：统一 API 网关（Auth、配额、限流、审计）、对话编排入口、与算法服务/豆包/Milvus 的粘合、SSE/WS 流式推送、运维与可观测

**子模块**：
- **Auth & 租户**：OIDC 登录、API Key、租户隔离（Header `X-Tenant-ID` + DB Scope）
- **限流/熔断/重试**：基于令牌桶（IP/用户/租户/路径粒度）、下游超时与重试策略（幂等路径）
- **SSE/WS 网关**：将 Python 算法服务的 Token Stream 转发到客户端；心跳与断点续传
- **编排层**：
  - Chat：接收 messages → 调用 Algo `/query` → 按 token 流式回传
  - Ingest：上传转存对象存储（本地/OSS/S3），下发解析任务到 Algo `/ingest`
  - Admin：数据集、索引状态、健康检查
- **观测**：
  - Tracing：W3C TraceContext → 贯穿前端、Go、Python、Milvus、Ark
  - Metrics：P50/P95、QPS、错误率、外呼耗时分解（检索、生成）
  - Logging：结构化 JSON，关联 TraceID/SpanID

**Gin 路由（示例）**：
```
POST /api/chat/stream         // SSE 返回
POST /api/ingest/upload
POST /api/ingest/url
GET  /api/ingest/tasks/:id
GET  /api/datasets            // 列表/状态
GET  /api/healthz
```

**与算法服务通信**：
- 协议：HTTP/JSON（MVP），可演进到 gRPC
- 超时：Query 30s、Ingest 300s；支持 `X-Request-ID`、`X-Tenant-ID`

---

## 3.3 算法（Python / LangChain）
**职责**：RAG 全流程与 LLM 调用，暴露轻量 API，被 Go 后端编排

**关键依赖**：`langchain`, `langchain-community`, `langchain-text-splitters`, `langchain-milvus`, `pymilvus`, `fastapi`/`litestar`, `uvicorn`, `tiktoken`（或 `dashscope-tokenizer` 替代）、`pydantic`

**服务接口**：
- `POST /ingest`：
  - 入参：{dataset_id, files[]|urls[], metadata, chunk_size, chunk_overlap, embedding_model}
  - 流程：
    1) 解析器（PDF/Doc/HTML/Markdown/纯文本）
    2) 清洗（去样式、去脚注/目录、正文抽取）
    3) 切片（`RecursiveCharacterTextSplitter` 约 500~800 tokens，overlap 50~120）
    4) 向量化（默认：豆包 Embedding *若未开通则回退 bge-m3/bge-large-zh*）
    5) 写入 Milvus：collection schema（`id`, `text`, `vector`, `source`, `chunk_id`, `doc_id`, `tenant_id`, `tags`, `updated_at`）
    6) 建索引/加载（HNSW / IVF_FLAT / HNSW_PQ，阈值：`score>=0.3` 可配置）
- `POST /query`：
  - 入参：{messages[], top_k=5, rerank=false|"bge-reranker", filters{tags,doc_id}, temperature, max_tokens}
  - 流程：
    1) 从 messages 提取最新 user query
    2) 检索（Milvus → `similarity_search_with_score`），可选多路召回 + MMR
    3) （可选）轻量重排（cross-encoder）
    4) Prompt 组装（系统指令 + 历史摘要 + 检索片段 + 引用格式）
    5) 调用豆包 ChatCompletions（流式）
    6) 返回：delta tokens + 引用元数据（chunk_id、source、score）
- `POST /summarize`（可选）：长文摘要、对话记忆压缩

**向量与索引**：
- 维度：与所选 embedding 模型一致（例如 bge-m3=1024）
- Metric：cosine 优先（文本相似），Milvus `IP` 或 `COSINE`
- 索引建议（MVP）：数据量 < 100 万 → **HNSW**；> 100 万或成本敏感 → **IVF_FLAT / HNSW_PQ**

**Prompt 基线（片段型问答）**：
```
系统：你是企业知识助手，只基于检索到的片段回答；无法从片段中找到依据时，明确告知“没有找到依据”，并提出可行的下一步。
用户问题：{query}
可用片段（带编号与来源）：
{context}
回答要求：
- 中文；给出引用编号，例如[1][3]
- 先结论再依据，简洁分点
- 如需步骤，使用 1/2/3 编号
```

---

# 四、数据与存储
- **Milvus**：
  - Collection 按租户/数据集分表或同表加 `tenant_id`/`dataset_id` 字段
  - 分区：按 `doc_id`/日期或 `tags`，便于冷热与 ACL
  - 索引：优先 HNSW（`M=16, efConstruction=200`），查询 `efSearch` 动态调参；或 IVF_FLAT（`nlist`≈√N）
- **对象存储**：原始文件与解析中间件（S3/OSS/本地磁盘）
- **关系型**：Postgres（用户、会话、配额、计费/配置信息）
- **缓存**：Redis（会话短期记忆、热问题缓存、embedding 结果缓存）

---

# 五、API 契约（MVP 草案）
## 5.1 对话流（SSE）
```
POST /api/chat/stream
Headers: Authorization: Bearer <token>
Body: {
  "conversation_id": "c_123"?,
  "messages": [
    {"role":"system","content":"..."}?,
    {"role":"user","content":"..."}
  ],
  "tools": [{"type":"search","args":{...}}]?,
  "top_k": 5,
  "temperature": 0.3
}
// SSE 事件：data: {"type":"delta","content":"...","refs":[{chunk_id,source,score}]}
```

## 5.2 文档入库
```
POST /api/ingest/upload  // Gin 接收，转发到 /ingest
POST /algo/ingest        // Python FastAPI
```

## 5.3 检索预览
```
GET /api/search?q=...&top_k=5&filters=...
```

---

# 六、部署与运维
- **镜像**：
  - `backend-gin`: 多阶段构建，启用 `GOMAXPROCS`, `GIN_MODE=release`
  - `algo-langchain`: `uvicorn --workers=2 --http h11 --loop uvloop`
- **K8s 建议**：
  - 每个服务 `HPA`（CPU 60%/自定义 QPS 指标），`PodDisruptionBudget`
  - **Milvus**：独立集群/云托管；SSD 优先；观测 Milvus `search_latency`、`nq`, `ef`
  - **密钥**：Ark API Key、DB、OIDC 密钥放入 `Secret`；使用 `ExternalSecret`/KMS
- **可观测**：
  - Tracing：前后端全链路（前端 header → Gin → Algo → Milvus/Ark）
  - Metrics：
    - 端到端 P95 < 800ms（纯对话，不含长检索）
    - RAG 回答首 token < 1.2s（SSE）
    - 检索召回率@5 ≥ 0.85（离线评测集）

---

# 七、安全与合规（MVP 必做）
- 租户隔离（逻辑级 ACL + 行级过滤），禁止跨租户检索
- 敏感词/PII 探测（上传与回答前置规则，违规打码/阻断）
- 审计日志（查询、上传、删除、管理操作）
- 速率限制与消费上限（按用户/租户/天）

---

# 八、演进路线
1) **Rerank 引入**：bge-reranker-base（可选）
2) **语义缓存**：query/embedding 缓存 + 近似匹配阈值
3) **工具调用**：FAQ/SQL/内网 API（增加 Tool Router 与参数提取）
4) **多模态**：图片/截图解析（文档 OCR），图片向量入库
5) **治理**：Prompt 版本管理、评测数据集与离线指标面板

---

# 九、目录结构（示例）
```
repo/
  frontend/            # Next.js
  backend/             # Go/Gin
    cmd/server
    internal/handler
    internal/service
    pkg/middleware
  algo/                # Python/LangChain/FastAPI
    app/api.py
    core/ingest.py
    core/retrieve.py
    core/prompt.py
  deploy/
    k8s/
    compose/
```

---

# 十、MVP 开发清单（TODO）
**前端**
- [ ] Chat UI（SSE）与消息列表（带复制、代码高亮）
- [ ] 引用片段 UI（编号/来源/打开展示）
- [ ] 文件上传（拖拽/多文件队列/进度条）
- [ ] 登录（OIDC/JWT）与基础角色（admin/user）
- [ ] 管理页：数据集列表、任务状态、简版统计

**后端（Go/Gin）**
- [ ] 路由与中间件（日志、恢复、限流、CORS、鉴权）
- [ ] `/api/chat/stream` SSE 转发与断点续传
- [ ] `/api/ingest/upload` 对象存储中转 + 调用算法 `/ingest`
- [ ] 多租户与配额（Header + DB Scope）
- [ ] OTel + Prometheus 指标与分布式追踪

**算法（Python/LangChain）**
- [ ] FastAPI 服务骨架（/ingest, /query）
- [ ] 文档解析器与切片策略（可配置 chunk）
- [ ] Milvus 集成（`langchain-milvus`，建表/建索引/加载）
- [ ] Embedding 适配（优先豆包，回退 bge-*）
- [ ] RAG Pipeline（召回→可选 MMR→Prompt→豆包流式）
- [ ] 简单评测脚本（召回率、答案 BLEU/ROUGE、人工标注采样）

**基础设施**
- [ ] Dockerfile/Compose（本地一键起）
- [ ] K8s 清单（HPA、PDB、Secret、ConfigMap）
- [ ] 环境配置（dev/staging/prod）与密钥管理（KMS/ExternalSecrets）

**质量与安全**
- [ ] 单元/集成测试（后端与算法）、契约测试（OpenAPI）
- [ ] 速率限制与消费上限、审计日志
- [ ] 敏感词规则与上传安全扫描（大小/类型/内容）

---

# 十一、关键参数建议（初版）
- 检索 TopK：5（MMR=0.5，可选）
- 相似度阈值：0.3（召回过滤）
- Chunk：600 tokens，overlap 80（按文档类型动态调）
- 索引：HNSW（M=16, efC=200, efS=64），规模上来后评估 IVF_FLAT/HNSW_PQ
- LLM：temperature 0.3，max_tokens 1024（MVP）

---

# 十二、风险与回退
- 豆包 Embedding 未开通/不稳定 → 回退 bge-m3/bge-large-zh
- Milvus 内存压力过大 → 启用分区/分表与 HNSW_PQ；降 `efSearch`
- Ark API 限流 → 后端排队与指数退避；预估配额&并发
- 长文/表格解析质量差 → 针对 PDF/HTML 引入专项解析器

---

# 十三、里程碑
- M1（第 1 周）：仓库搭建、SSE 通路、最小对话通
- M2（第 2~3 周）：Ingest 全链路、Milvus 写入与检索、RAG 首答
- M3（第 4 周）：管理后台、观测、配额/限流、安全最小集
- M4（第 5 周）：评测与调参、稳定性与成本优化、试点上线
