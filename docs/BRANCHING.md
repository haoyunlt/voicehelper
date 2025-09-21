# 分支规划（Trunk-Based）

## 0. 命名规范
- 功能分支：`feat/<scope>-<short>` （例：`feat/gateway-sse`）
- 修复分支：`fix/<scope>-<short>`
- 运维分支：`chore/<topic>`、`docs/<topic>`、`perf/<topic>`
- 版本标签：`v1.0.0`（MVP），遵循 SemVer；发布分支可选 `release/v1.0`

## 1. 合并策略
- 所有 PR → `main`（受保护）  
- **必须通过检查**：`lint`、`unit`、`contract`（OpenAPI/SSE/WS 回放）、`e2e`、`k6-smoke`  
- 至少 1 名 Code Owner 评审 + 1 名跨域评审（FE/BE/DS）  
- 禁止直接 push `main`

## 2. Feature Flags
- `VOICE_MP`（小程序入口开关）
- `MCP_WRITE`（工具写操作开关）
- `AGENT_EVENTS_UI`（前端事件可视化）

## 3. PR 规范（Conventional Commits）
- `feat: gateway sse stream with cancel`
- `fix: ws binary frame parsing edge case`
- `docs: add voice ws schema`
- `perf: reduce tts first audio latency`

## 4. 回滚与热修
- 回滚：`revert` PR + `helm rollback`（保留 2 版历史）  
- 热修：`hotfix/<issue>` → PR → 通过同样 Gate 后合 `main` 并打 tag

## 5. 目录与 CODEOWNERS（示例）
```
/frontend     @owner-fe
/miniprogram  @owner-mp
/backend      @owner-be
/algo         @owner-ds
/deploy       @owner-devops
/docs         @owner-pm
```
在 `.github/CODEOWNERS` 中配置；PR 必须包含对应域的 Owner。

## 6. CI 工作流（概要）
- `lint.yml`：go vet / golangci-lint；ruff+mypy；eslint+tsc  
- `test.yml`：Go/Py 单测；Py 集成测（Milvus docker）  
- `contract.yml`：schemathesis（OpenAPI）+ WS/SSE 回放  
- `e2e.yml`：Playwright（Web）；手动/云测（MP）  
- `perf.yml`：k6 烟测（文本/语音混合）  
- `deploy.yml`：Helm 部署 `stg`；人工批准后金丝雀到 `prod`

## 7. 版本里程碑（与 Roadmap 对齐）
- v1.0.0：MVP（本文 TODO 范围）  
- v1.1.0：RAG/运营增强（知识库管理、用量榜单）  
- v1.2.0：成本与性能（语义缓存、批量 TTS 优化）  
- v1.3.0：语音 P1（整段 TTS）  
- v1.4.0：语音 P2（流式 + barge‑in 强化、MCP 写操作）