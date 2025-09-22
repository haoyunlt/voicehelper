# 重构后的项目目录结构

## 整体架构

```
repo/
├── frontend/                    # Node.js / Next.js
│   └── src/
│       ├── api/                 # TS SDK (SSE/WS)
│       │   ├── base.ts
│       │   ├── chat.ts
│       │   └── voice.ts
│       ├── audio/               # AudioWorklet & players
│       │   ├── base.ts
│       │   └── pcm16.ts
│       ├── pages/               # Next.js页面
│       │   ├── analytics/
│       │   ├── chat/
│       │   ├── datasets/
│       │   ├── globals.css
│       │   ├── layout.tsx
│       │   └── page.tsx
│       └── components/          # React组件
│           ├── chat/
│           ├── dialog/
│           ├── ui/
│           └── voice/
├── backend/                     # Go / Gin
│   ├── cmd/
│   │   └── gateway/
│   │       └── main.go          # 网关入口
│   └── internal/
│       ├── handler/             # Chat/Voice/Agent handlers
│       │   ├── auth.go
│       │   ├── chat.go
│       │   ├── dataset.go
│       │   ├── handler.go
│       │   ├── ingest.go
│       │   ├── integration.go
│       │   ├── security_admin.go
│       │   └── voice.go
│       ├── middleware/          # Auth/RateLimit/Tracing
│       │   └── zero_trust.go
│       ├── ssews/               # 共用SSE/WS封装(父类思想)
│       │   ├── sse.go
│       │   ├── stream.go
│       │   └── ws.go
│       ├── contracts/           # DTO/Envelope/Error
│       │   ├── response.go
│       │   └── validation.go
│       └── service/             # 调用algo服务的client
│           ├── chat.go
│           ├── model_router_service.go
│           └── service.go
├── algo/                        # Python / FastAPI / LangGraph
│   ├── app/
│   │   ├── api.py               # /query /voice /agent /ingest
│   │   ├── main.py
│   │   └── v2_api.py
│   ├── core/
│   │   ├── base/                # 抽象父类 & Mixins
│   │   │   ├── __init__.py
│   │   │   ├── mixins.py
│   │   │   ├── protocols.py
│   │   │   └── runnable.py
│   │   ├── graph/               # LangGraph图/节点（子类化）
│   │   │   ├── __init__.py
│   │   │   └── chat_voice.py
│   │   ├── tools/               # LangChain工具子类
│   │   │   └── [4 files]
│   │   ├── asr_tts/             # ASR/TTS 适配器子类
│   │   │   ├── __init__.py
│   │   │   ├── azure.py
│   │   │   ├── base.py
│   │   │   └── openai.py
│   │   ├── rag/                 # Retriever/Chunker/Embedder
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── bge_faiss_retriever.py
│   │   │   ├── embedder_bge.py
│   │   │   ├── ingest_faiss.py
│   │   │   ├── metrics.py
│   │   │   └── retriever_faiss.py
│   │   └── memory/              # 会话/检查点
│   ├── adapters/                # 具体供应商接入（子类）
│   │   └── __init__.py
│   └── settings.py
├── deploy/                      # Helm/compose
│   ├── compose/
│   ├── config/
│   ├── database/
│   ├── helm/
│   ├── k8s/
│   ├── monitoring/
│   ├── scripts/
│   └── ssl/
└── docs/
    ├── openapi.yaml
    └── design-latest.md
```

## 主要变更

### Frontend 重构
- 将现有的 `src/api/`、`src/audio/` 保持原位
- 将 `app/` 目录内容移动到 `src/pages/`
- 将 `components/` 移动到 `src/components/`
- 保持现有的 hooks、lib 等目录结构

### Backend 重构
- 将 `cmd/server/main.go` 移动到 `cmd/gateway/main.go`
- 保持现有的 `internal/` 结构，但增加了逻辑分组
- 将 `pkg/common/` 内容移动到 `internal/contracts/`
- 保持现有的 handler、middleware、service 结构

### Algo 重构
- 保持现有的 `core/` 结构，但增加了更清晰的分组
- 现有的 `core/base/`、`core/graph/`、`core/tools/`、`core/asr_tts/`、`core/rag/` 已存在
- 新增 `adapters/` 目录用于具体供应商接入
- 保持 `app/` 作为 FastAPI 入口

## 兼容性说明

此重构保持了现有文件的位置，主要是创建了新的目录结构并建立了符合样例的组织方式。大部分现有代码无需修改import路径，因为：

1. Frontend的核心文件（src/api、src/audio）保持原位
2. Backend的内部结构基本保持不变
3. Algo的核心模块结构已经符合要求

## 下一步

1. 根据需要调整具体的import路径
2. 将相关文件移动到对应的新目录
3. 更新配置文件中的路径引用
4. 更新文档和部署脚本
