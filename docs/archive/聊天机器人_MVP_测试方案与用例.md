# 聊天机器人（豆包 + Milvus + LangChain/Python + Go/Gin + Node.js）测试方案与用例

> 版本：v1.0（MVP 对齐）  
> 目标读者：QA / 后端 / 算法 / 前端 / 运维  
> 关联文档：系统设计方案（MVP）

---

## 1. 测试目标与质量门槛（MVP）
**业务目标**
- 核心对话与 RAG 闭环可用：上传 → 解析 → 切片 → 向量化 → Milvus 入库 → 检索 → 生成 → 引用展示

**质量门槛（可量化）**
- **功能**：关键路径通过率 ≥ 98%（自动化 + 人工）
- **性能**：
  - 端到端 P95（纯对话，无检索） < **800ms**
  - RAG 首 Token 时间 < **1.2s**；整轮 P95 < **2.8s**
  - 稳态 QPS（对话）≥ **20**（单副本，参考值，按实际资源校准）
- **检索质量**：离线评测集 Recall@5 ≥ **0.85**
- **稳健性**：断线/超时/限流 等异常均有可恢复策略，且无数据错乱
- **安全**：租户隔离零越权、上传安全检查、速率限制生效、审计完整

---

## 2. 测试范围与分层
**分层策略（测试金字塔）**
1. **单元测试**（Python / Go / 前端）：最小逻辑验证（覆盖率目标：后端/算法 ≥ 70%）
2. **服务集成测试**（Go↔Python、Python↔Milvus、Go↔Ark）：契约与错误处理
3. **端到端 E2E**（浏览器→后端→算法→Milvus/Ark）：SSE 流、引用展示、鉴权等
4. **非功能**：性能、可靠性、可观测、安全/合规、成本与配额

**测试环境**
- **dev**：本地/CI，Milvus（standalone）、Ark Sandbox、最小副本
- **staging**：接近生产参数，真实 Milvus 集群，灰度 Ark Key，开启 OTel/Prom
- **prod**：仅灰度流量/只读巡检用例

---

## 3. 测试数据与准备
- **基准文档集**：中/英/表格/长文/FAQ/规范类（不少于 30 篇）
- **切片策略**：600 tokens，overlap 80（按文类校准一份备选策略 300/50 与 800/120）
- **检索评测集（问答对）**：≥ 200 条；标注正确片段 ID + 参考答案
- **账号/租户**：`tenant_a` / `tenant_b`；`admin` / `user` 角色各 2 个
- **密钥**：Ark API Key（灰度）、对象存储/DB/Milvus 访问凭据（staging 独立）

---

## 4. 工具链与度量
- **单元**：`pytest`, `unittest`, `go test`, `testify`, `httptest`, `supertest`/`vitest`
- **契约**：`OpenAPI` + `schemathesis`/`dredd`
- **E2E**：`Playwright`（首选）/`Cypress`（次选）
- **性能**：`k6` 或 `vegeta`（SSE 支持用 `k6` WebSocket/SSE 扩展）
- **混沌/故障注入**：`toxiproxy`, `pumba`（网络/CPU/IO），Milvus `load/release` 操作模拟
- **观测校验**：`Prometheus` 指标、OTel Trace（链路完整性、span 属性）

---

## 5. 测试用例（核心功能）

### 5.1 文档入库（/api/ingest/upload → /algo/ingest）
| 用例ID | 场景 | 前置 | 步骤 | 期望结果 |
|---|---|---|---|---|
| ING-001 | 上传 PDF 成功 | 登录 admin | 上传 5MB 正常 PDF；查询 task 状态 | task=done；切片数>0；Milvus 中可检索到；索引已 load |
| ING-002 | 不支持类型 | 登录 admin | 上传 `.exe`/超大文件>100MB | 阻断，返回 4xx + 明确错误码（`ERR_FILE_TYPE`/`ERR_SIZE_LIMIT`）；审计记录 |
| ING-003 | HTML 解析 | 登录 admin | 上传复杂 HTML（含目录/脚注） | 提取正文，去噪成功；切片符合策略（平均长度±20%） |
| ING-004 | 多租户隔离 | 两租户 | `tenant_a` 上传同名文档；`tenant_b` 检索 | `tenant_b` 无法看到 `tenant_a` 文档/向量 |
| ING-005 | 索引构建失败回退 | Milvus 降级 | 上传触发建索引失败 | 返回 task=failed + 可重试；日志含 root cause；系统可继续其他任务 |

### 5.2 检索/重排（/api/search, /algo/query）
| 用例ID | 场景 | 步骤 | 期望 |
|---|---|---|---|
| RET-001 | TopK=5 基础检索 | 以标注问题集执行 | 标注命中率 Recall@5 ≥0.85 |
| RET-002 | MMR 生效 | 设置 `use_mmr=true` | 返回片段相似度更分散（去冗余），答案一致性不下降 |
| RET-003 | 过滤器 | `filters: {tags:["policy"]}` | 返回仅来自 `policy` 标签文档 |
| RET-004 | 空召回 | 查询冷门问题 | 返回“未找到依据”模板话术；不胡编 |
| RET-005 | 重排可选 | 开启 `rerank=bge-reranker` | Top1 相关性提升，离线集 MAP 提升≥5% |

### 5.3 对话生成（/api/chat/stream）
| 用例ID | 场景 | 步骤 | 期望 |
|---|---|---|---|
| CHAT-001 | 纯对话 | 发送闲聊 | 首 token <1s；全响应 P95<800ms |
| CHAT-002 | RAG 对话 | 询问文档事实 | 回答含引用 [1][2]；点击引用可展开原片段 |
| CHAT-003 | SSE 断点恢复 | 传输中断后重发 same request-id | 从最近 token 继续；无重复块/错序 |
| CHAT-004 | 上下文记忆（短期） | 连续三轮问题 | 第三轮可正确关联上一轮结论；token 不溢出 |
| CHAT-005 | 限流与配额 | 压测超限 | 返回 429；`Retry-After` 正确；审计与 metrics 计数 +1 |

### 5.4 鉴权/租户/角色
| 用例ID | 场景 | 步骤 | 期望 |
|---|---|---|---|
| AUTH-001 | 未登录访问 | 直接调用受限接口 | 401 + 统一错误体 |
| AUTH-002 | 普通用户上传 | user 角色上传 | 403（仅 admin 可上传） |
| AUTH-003 | 租户隔离 | 切换 `X-Tenant-ID` | 资源范围随租户切换；越权无数据 |

### 5.5 前端交互
| 用例ID | 场景 | 步骤 | 期望 |
|---|---|---|---|
| UI-001 | 流式渲染 | SSE 渲染 3 条消息 | 无抖动；滚动定位到最新；代码块高亮正常 |
| UI-002 | 上传进度 | 上传 3 个文件 | 正确显示进度与失败原因；可重试 |
| UI-003 | 反馈与埋点 | 点赞/点踩 | 事件上报成功；后端计数 +1；可在 /analytics 查看 |

---

## 6. 非功能测试

### 6.1 性能与容量（k6/vegeta）
**场景**：
1. **基线对话**：R=20 并发，60s 稳态 → P95 < 800ms
2. **RAG 对话**：R=10 并发（含 Milvus），300s 稳态 → 首 token <1.2s，P95 < 2.8s
3. **入库并发**：并行 5 路 ingest，平均文档 5MB → 任务平均完成 < 90s；Milvus search P95 不显著劣化（<+15%）

**报告**：响应分布、错误率、资源（CPU/内存/IO）、Milvus 指标（`search_qps`, `search_latency`, `efSearch`）

### 6.2 可靠性与恢复
- **网络抖动**：算法服务延迟 200~1000ms、5% 丢包 → UI 仍可流式；后端无 goroutine 堆积
- **下游超时**：Ark 超时 3s → 后端重试 ≤1 次；返回可读错误并记录审计
- **Milvus load/release**：索引释放后自动 load；检索先行失败后重试成功

### 6.3 安全与合规
- 上传安全：黑名单后缀/内容型检查（宏/脚本），超限阻断
- 注入类：prompt 注入/HTML 注入/路径注入 → 服务器端严格转义/白名单
- 速率限制：IP/用户/租户维度令牌桶；暴力请求在 10s 内触发 429
- 审计：登录、上传、删除、重大配置变更均有审计记录

### 6.4 可观测验证
- 随机挑 20 条 E2E 请求，检查 Trace 是否贯穿（前端 → Gin → Algo → Milvus/Ark）
- 指标齐全：`http_server_requests_seconds_*`、`chat_stream_first_token_seconds`、`milvus_search_latency`

---

## 7. 检索质量评测（RAG 专项）
**离线评测流程**
1. 以标注问答集生成检索请求
2. 统计 Recall@K（K=1/3/5）、nDCG@K、MAP
3. 选取 50 条样本做人审（事实性/完整性/引用正确）
4. 形成基线：更换切片/索引/重排参数进行 A/B（每次单变量）

**通过阈值（MVP）**
- Recall@5 ≥ 0.85；引用正确率 ≥ 0.9；明显幻觉率 ≤ 3%

---

## 8. 契约与错误码

**统一错误体（示例）**
```json
{
  "error": {
    "code": "ERR_RATE_LIMIT",
    "message": "Too many requests",
    "request_id": "r-20250921-xxx"
  }
}
```

**常见错误码**
- `ERR_UNAUTHORIZED` / `ERR_FORBIDDEN` / `ERR_RATE_LIMIT` / `ERR_TIMEOUT`
- `ERR_FILE_TYPE` / `ERR_FILE_SIZE` / `ERR_INGEST_FAILED`
- `ERR_SEARCH_UNAVAILABLE` / `ERR_LLM_BACKEND`

---

## 9. 测试自动化与CI
- PR 必跑：单元 + 契约 + 快速集成（Milvus 可用 docker service）
- 夜间任务：E2E 冒烟 + 性能 10 分钟小压
- 质量门槛：覆盖率阈值、lint、依赖漏洞扫描（`trivy`, `npm audit`）

---

## 10. 测试清单（Checklist）
- [ ] 关键路径：上传→检索→生成→引用
- [ ] 多租户隔离与权限
- [ ] SSE 流与断点恢复
- [ ] 性能 P95 与首 Token
- [ ] Milvus 查询/索引参数回归
- [ ] Ark 后端超时/限流/降级回退
- [ ] 安全：上传/注入/速率限制/审计
- [ ] 观测：Trace 贯穿、指标齐全、日志结构化

---

## 11. 附录：示例请求

**对话（SSE）**
```http
POST /api/chat/stream
Authorization: Bearer <token>
X-Tenant-ID: tenant_a
Content-Type: application/json

{
  "conversation_id": "c_001",
  "messages": [{"role":"user","content":"根据公司请假制度，事假流程是什么？"}],
  "top_k": 5,
  "temperature": 0.3
}
```

**算法检索（/algo/query）**
```json
{
  "messages": [{"role":"user","content":"里程碑M2的交付范围？"}],
  "top_k": 5,
  "rerank": false,
  "filters": {"tags": ["project"]}
}
```

**入库（/algo/ingest）**
```json
{
  "dataset_id": "ds_2025_hr",
  "urls": ["https://intranet/policy/leave.html"],
  "chunk_size": 600,
  "chunk_overlap": 80,
  "embedding_model": "doubao-embedding" 
}
```

---

> 注：上线前需完成演练：1) Ark 限流/超时注入；2) Milvus 索引 release/reload；3) SSE 网络抖动；4) 多租户穿越攻击模拟。
