# VoiceHelper V2 架构重构计划 - 父类/子类设计模式

## 概述

基于父类（抽象基类/协议）沉淀公共逻辑，子类在清晰扩展点上覆盖/增强的设计原则，对整个系统进行架构重构。采用 Mixin 模式解耦横切能力，以 Pydantic + JSONSchema、interface 约束接口契约。

**技术约定**：
- Python ≥3.11，Pydantic v2
- Go ≥1.22（Gin）
- Node ≥18（Next.js）

---

## 1. 目录骨架设计

```
voicehelper/
├── frontend/                    # Node.js / Next.js
│   ├── src/
│   │   ├── api/                 # TS SDK (SSE/WS)
│   │   │   ├── base.ts          # 父类客户端抽象
│   │   │   ├── chat.ts          # ChatSSEClient 子类
│   │   │   └── voice.ts         # VoiceWSClient 子类
│   │   ├── audio/               # AudioWorklet & players
│   │   │   ├── base.ts          # BaseAudioProcessor 父类
│   │   │   ├── pcm16.ts         # PCM16Processor 子类
│   │   │   └── player.ts        # BasePlayer & PCMChunkPlayer
│   │   ├── pages/
│   │   └── components/
│   └── package.json
├── backend/                     # Go / Gin
│   ├── cmd/gateway/main.go
│   ├── internal/
│   │   ├── handler/             # Chat/Voice/Agent handlers
│   │   │   ├── base.go          # BaseHandler 父类
│   │   │   ├── chat.go          # ChatHandler 子类
│   │   │   └── voice.go         # VoiceHandler 子类
│   │   ├── middleware/          # Auth/RateLimit/Tracing
│   │   ├── ssews/               # SSE/WS封装父类
│   │   │   ├── stream.go        # StreamWriter 接口
│   │   │   ├── sse.go           # SSE 实现
│   │   │   └── ws.go            # WebSocket 实现
│   │   ├── contracts/           # DTO/Envelope/Error
│   │   └── service/             # 调用algo服务的client
│   └── go.mod
├── algo/                        # Python / FastAPI / LangGraph
│   ├── app/
│   │   └── api.py               # /query /voice /agent /ingest
│   ├── core/
│   │   ├── base/                # 抽象父类 & Mixins
│   │   │   ├── __init__.py
│   │   │   ├── runnable.py      # 核心父类定义
│   │   │   ├── mixins.py        # RetryableMixin, ObservableMixin
│   │   │   ├── protocols.py     # AsrStream, TtsStream 协议
│   │   │   └── tools.py         # BaseTool 父类
│   │   ├── graph/               # LangGraph图/节点
│   │   │   ├── base.py          # BaseAgentGraph 父类
│   │   │   ├── chat_voice.py    # ChatVoiceAgentGraph 子类
│   │   │   └── nodes/           # 各种节点实现
│   │   ├── tools/               # LangChain工具子类
│   │   │   ├── fetch.py         # FetchTool 子类
│   │   │   ├── fs_read.py       # FsReadTool 子类
│   │   │   └── github_read.py   # GithubReadTool 子类
│   │   ├── asr_tts/             # ASR/TTS 适配器子类
│   │   │   ├── base.py          # 适配器父类
│   │   │   ├── openai.py        # OpenAI 适配器子类
│   │   │   └── azure.py         # Azure 适配器子类
│   │   ├── rag/                 # Retriever/Chunker/Embedder
│   │   │   ├── base.py          # BaseRetriever 父类
│   │   │   ├── milvus.py        # MilvusRetriever 子类
│   │   │   └── chunker.py       # 文档切分器
│   │   └── memory/              # 会话/检查点
│   ├── adapters/                # 具体供应商接入（子类）
│   ├── requirements.txt
│   └── settings.py
├── deploy/                      # Helm/compose
└── docs/
    ├── openapi.yaml
    └── design-latest.md
```

---

## 2. 算法服务（Python）- 父类/子类设计

### 2.1 抽象父类与协议定义

#### `algo/core/base/protocols.py`
```python
from typing import Iterable, Iterator, Protocol, Callable, Optional

class StreamCallback(Protocol):
    """流式回调协议"""
    def __call__(self, event: str, payload: dict) -> None: ...

class AsrStream(Protocol):
    """ASR 流式处理协议"""
    def start(self, sr: int, codec: str, lang: str) -> str: ...
    def feed(self, seq: int, chunk: bytes) -> None: ...
    def stop(self) -> None: ...
    def on_partial(self, cb: Callable[[int, str], None]) -> None: ...
    def on_final(self, cb: Callable[[int, str, float], None]) -> None: ...

class TtsStream(Protocol):
    """TTS 流式合成协议"""
    def synthesize(self, text_iter: Iterable[str]) -> Iterator[bytes]: ...
    def cancel(self, request_id: str) -> None: ...
```

#### `algo/core/base/mixins.py`
```python
from typing import Callable, Any, Optional
import logging
import time
from functools import wraps

class RetryableMixin:
    """重试能力 Mixin"""
    max_retries: int = 1
    retry_delay: float = 0.1
    
    def _retry(self, fn: Callable[[], Any]) -> Any:
        """重试封装"""
        for i in range(self.max_retries + 1):
            try:
                return fn()
            except Exception as e:
                if i == self.max_retries:
                    raise
                time.sleep(self.retry_delay * (2 ** i))

class ObservableMixin:
    """可观测能力 Mixin"""
    def emit(self, cb: Optional[StreamCallback], event: str, payload: dict):
        """发送事件"""
        if cb:
            cb(event, payload)
    
    def _log_performance(self, operation: str):
        """性能日志装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    logging.info(f"{operation} completed in {duration:.3f}s")
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logging.error(f"{operation} failed after {duration:.3f}s: {e}")
                    raise
            return wrapper
        return decorator

class CacheableMixin:
    """缓存能力 Mixin"""
    _cache: dict = {}
    cache_ttl: int = 300  # 5分钟
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        import hashlib
        key_str = f"{args}_{sorted(kwargs.items())}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _cached_call(self, fn: Callable, *args, **kwargs):
        """缓存调用"""
        cache_key = self._get_cache_key(*args, **kwargs)
        now = time.time()
        
        if cache_key in self._cache:
            cached_time, result = self._cache[cache_key]
            if now - cached_time < self.cache_ttl:
                return result
        
        result = fn(*args, **kwargs)
        self._cache[cache_key] = (now, result)
        return result
```

#### `algo/core/base/runnable.py`
```python
from typing import Iterator, Optional, Any, Dict, List
from pydantic import BaseModel, Field
from .mixins import RetryableMixin, ObservableMixin, CacheableMixin
from .protocols import StreamCallback

class BaseTool(BaseModel, RetryableMixin, ObservableMixin):
    """工具基类"""
    name: str
    description: str
    args_schema: dict = Field(default_factory=dict)  # JSONSchema
    
    def validate_args(self, **kwargs) -> dict:
        """参数校验"""
        # TODO: 基于 args_schema 进行 JSONSchema 校验
        return kwargs
    
    def run(self, **kwargs) -> dict:
        """执行工具 - 子类必须实现"""
        raise NotImplementedError
    
    def run_with_callback(self, cb: Optional[StreamCallback] = None, **kwargs) -> dict:
        """带回调的执行"""
        self.emit(cb, "tool_start", {"tool": self.name, "args": kwargs})
        try:
            result = self._retry(lambda: self.run(**kwargs))
            self.emit(cb, "tool_result", {"tool": self.name, "result": result})
            return result
        except Exception as e:
            self.emit(cb, "tool_error", {"tool": self.name, "error": str(e)})
            raise

class BaseRetriever(BaseModel, RetryableMixin, ObservableMixin, CacheableMixin):
    """检索器基类"""
    top_k: int = 5
    
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """检索 - 子类必须实现"""
        raise NotImplementedError
    
    def retrieve_with_callback(self, query: str, cb: Optional[StreamCallback] = None, **kwargs) -> List[Dict[str, Any]]:
        """带回调的检索"""
        self.emit(cb, "retrieve_start", {"query": query})
        try:
            # 尝试从缓存获取
            result = self._cached_call(self.retrieve, query, **kwargs)
            self.emit(cb, "retrieve_result", {"query": query, "count": len(result)})
            return result
        except Exception as e:
            self.emit(cb, "retrieve_error", {"query": query, "error": str(e)})
            raise

class BaseAgentGraph(BaseModel, ObservableMixin):
    """Agent 图基类"""
    retriever: BaseRetriever
    tools: List[BaseTool] = []
    
    def stream(self, query: str, *, cb: Optional[StreamCallback] = None) -> Iterator[dict]:
        """统一流式接口 - 子类必须实现"""
        raise NotImplementedError
    
    def _get_tool_by_name(self, name: str) -> Optional[BaseTool]:
        """根据名称获取工具"""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
```

### 2.2 子类实现

#### `algo/core/rag/milvus.py`
```python
from typing import List, Dict, Any, Optional
from ..base.runnable import BaseRetriever
from ..base.protocols import StreamCallback

class MilvusRetriever(BaseRetriever):
    """Milvus 向量检索器"""
    collection_name: str
    host: str = "localhost"
    port: int = 19530
    use_mmr: bool = True
    mmr_lambda: float = 0.5
    
    def __init__(self, **data):
        super().__init__(**data)
        # TODO: 初始化 Milvus 连接
    
    def retrieve(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """执行向量检索"""
        filters = kwargs.get('filters')
        top_k = kwargs.get('top_k', self.top_k)
        
        # TODO: 实现 Milvus 检索逻辑
        # 1. 文本嵌入
        # 2. 向量检索
        # 3. MMR 重排（如果启用）
        # 4. 格式化输出
        
        return [
            {
                "content": "示例内容",
                "score": 0.95,
                "source": "doc1.pdf",
                "url": "https://example.com/doc1.pdf"
            }
        ]
```

#### `algo/core/tools/fetch.py`
```python
from typing import Dict, Any
import httpx
from ..base.runnable import BaseTool

class FetchTool(BaseTool):
    """URL 获取工具"""
    name: str = "fetch_url"
    description: str = "GET 一个URL并返回文本内容"
    args_schema: dict = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "format": "uri"},
            "timeout": {"type": "number", "default": 5}
        },
        "required": ["url"]
    }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """执行URL获取"""
        validated_args = self.validate_args(**kwargs)
        url = validated_args["url"]
        timeout = validated_args.get("timeout", 5)
        
        def _fetch():
            with httpx.Client() as client:
                resp = client.get(url, timeout=timeout)
                resp.raise_for_status()
                return resp.text[:4000]  # 限制长度
        
        text = self._retry(_fetch)
        return {
            "text": text,
            "url": url,
            "status": "success"
        }
```

#### `algo/core/graph/chat_voice.py`
```python
from typing import Iterator, Dict, Any, Optional
from ..base.runnable import BaseAgentGraph
from ..base.protocols import StreamCallback

class ChatVoiceAgentGraph(BaseAgentGraph):
    """聊天语音 Agent 图"""
    
    def stream(self, query: str, *, cb: Optional[StreamCallback] = None) -> Iterator[Dict[str, Any]]:
        """流式处理"""
        try:
            # 1. 意图识别
            self.emit(cb, "agent_plan", {"step": "intent", "query": query})
            intent = self._analyze_intent(query)
            yield {"event": "intent", "data": intent}
            
            # 2. 检索（如需要）
            if intent.get("need_retrieval"):
                self.emit(cb, "agent_step", {"step": "retrieve", "query": query})
                docs = self.retriever.retrieve_with_callback(query, cb)
                yield {"event": "retrieve", "data": {"docs": docs}}
            
            # 3. 计划生成
            self.emit(cb, "agent_step", {"step": "plan"})
            plan = self._generate_plan(query, intent)
            yield {"event": "plan", "data": plan}
            
            # 4. 工具执行
            for step in plan.get("steps", []):
                if step.get("tool"):
                    tool = self._get_tool_by_name(step["tool"])
                    if tool:
                        result = tool.run_with_callback(cb, **step.get("args", {}))
                        yield {"event": "tool_result", "data": {"tool": step["tool"], "result": result}}
            
            # 5. 综合回答
            self.emit(cb, "agent_step", {"step": "synthesize"})
            answer = self._synthesize_answer(query, plan)
            yield {"event": "answer", "data": answer}
            
            # 6. TTS（如需要）
            if intent.get("need_tts"):
                self.emit(cb, "agent_step", {"step": "tts"})
                for audio_chunk in self._text_to_speech(answer["text"]):
                    yield {"event": "audio", "data": audio_chunk}
            
            self.emit(cb, "agent_summary", {"status": "completed"})
            
        except Exception as e:
            self.emit(cb, "agent_error", {"error": str(e)})
            yield {"event": "error", "data": {"error": str(e)}}
    
    def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """意图分析"""
        # TODO: 实现意图分析逻辑
        return {"need_retrieval": True, "need_tts": False}
    
    def _generate_plan(self, query: str, intent: Dict[str, Any]) -> Dict[str, Any]:
        """生成执行计划"""
        # TODO: 实现计划生成逻辑
        return {"steps": []}
    
    def _synthesize_answer(self, query: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """综合答案"""
        # TODO: 实现答案综合逻辑
        return {"text": "这是一个示例回答", "references": []}
    
    def _text_to_speech(self, text: str) -> Iterator[bytes]:
        """文本转语音"""
        # TODO: 实现 TTS 逻辑
        yield b"audio_chunk_placeholder"
```

### 2.3 功能清单（算法侧）

- [ ] **A-1 父类与Mixin落地**
  - **任务**：实现 `BaseTool/BaseRetriever/BaseAgentGraph/RetryableMixin/ObservableMixin/CacheableMixin`
  - **DoD**：可被子类直接 import，单测覆盖 >80%，文档完整
  - **文件**：`algo/core/base/` 目录下所有文件

- [ ] **A-2 MilvusRetriever 子类**
  - **任务**：实现向量检索，支持 MMR 重排和语义去重
  - **DoD**：RAG 压测 P95 <120ms（10万向量），Recall@5≥0.85
  - **文件**：`algo/core/rag/milvus.py`

- [ ] **A-3 工具三件套子类**
  - **任务**：实现 `FetchTool`, `FsReadTool`, `GithubReadTool`
  - **DoD**：三条端到端用例通过（Agent 自动决定是否调用）
  - **文件**：`algo/core/tools/` 目录下各工具文件

- [ ] **A-4 ChatVoiceAgentGraph.stream()**
  - **任务**：实现流式 Agent 图，支持事件回调
  - **DoD**：能边生成边发（delta），能在 TTS 阶段接受取消
  - **文件**：`algo/core/graph/chat_voice.py`

- [ ] **A-5 ASR/TTS 适配器**
  - **任务**：实现 `AsrAdapter`, `TtsAdapter` 协议适配器
  - **DoD**：`start→feed→final→synthesize` 完整闭环，TTS 首音 <500ms
  - **文件**：`algo/core/asr_tts/` 目录下各适配器文件

- [ ] **A-6 FastAPI 接口**
  - **任务**：实现 SSE 和 WebSocket 接口
  - **DoD**：SSE/WS 回放脚本通过，支持取消和背压
  - **文件**：`algo/app/api.py`

**运行命令（算法）**：
```bash
cd algo && uvicorn app.api:app --host 0.0.0.0 --port 8070 --reload
```

---

## 3. 网关（Go/Gin）- 父类封装设计

### 3.1 SSE/WS 父类封装

#### `backend/internal/ssews/stream.go`
```go
package ssews

import (
    "encoding/json"
    "net/http"
)

// 统一消息信封
type Envelope struct {
    Meta  map[string]interface{} `json:"meta,omitempty"`
    Data  interface{}            `json:"data,omitempty"`
    Error *ErrorInfo             `json:"error,omitempty"`
}

type ErrorInfo struct {
    Code    string `json:"code"`
    Message string `json:"message"`
}

// 流式写入器接口
type StreamWriter interface {
    WriteEvent(event string, payload interface{}) error
    WriteError(code, message string) error
    Close() error
}

// 基础流处理器
type BaseStreamHandler struct {
    Writer StreamWriter
    TraceID string
    TenantID string
}

func (h *BaseStreamHandler) WriteEnvelope(event string, data interface{}) error {
    envelope := Envelope{
        Meta: map[string]interface{}{
            "trace_id":  h.TraceID,
            "tenant_id": h.TenantID,
            "timestamp": time.Now().Unix(),
        },
        Data: data,
    }
    return h.Writer.WriteEvent(event, envelope)
}
```

#### `backend/internal/ssews/sse.go`
```go
package ssews

import (
    "encoding/json"
    "fmt"
    "net/http"
)

type SSEWriter struct {
    w      http.ResponseWriter
    flusher http.Flusher
    closed bool
}

func NewSSEWriter(w http.ResponseWriter) *SSEWriter {
    flusher, ok := w.(http.Flusher)
    if !ok {
        return nil
    }
    
    w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")
    w.Header().Set("Access-Control-Allow-Origin", "*")
    
    return &SSEWriter{w: w, flusher: flusher}
}

func (s *SSEWriter) WriteEvent(event string, payload interface{}) error {
    if s.closed {
        return fmt.Errorf("writer closed")
    }
    
    data, err := json.Marshal(payload)
    if err != nil {
        return err
    }
    
    fmt.Fprintf(s.w, "event: %s\n", event)
    fmt.Fprintf(s.w, "data: %s\n\n", data)
    s.flusher.Flush()
    
    return nil
}

func (s *SSEWriter) WriteError(code, message string) error {
    return s.WriteEvent("error", ErrorInfo{Code: code, Message: message})
}

func (s *SSEWriter) Close() error {
    s.closed = true
    return nil
}
```

### 3.2 Handler 子类实现

#### `backend/internal/handler/base.go`
```go
package handler

import (
    "context"
    "github.com/gin-gonic/gin"
    "voicehelper/internal/ssews"
)

type BaseHandler struct {
    AlgoServiceURL string
}

func (h *BaseHandler) extractTraceInfo(c *gin.Context) (traceID, tenantID string) {
    traceID = c.GetHeader("X-Trace-ID")
    if traceID == "" {
        traceID = generateTraceID()
    }
    
    tenantID = c.GetHeader("X-Tenant-ID")
    if tenantID == "" {
        tenantID = "default"
    }
    
    return
}

func (h *BaseHandler) createStreamHandler(writer ssews.StreamWriter, traceID, tenantID string) *ssews.BaseStreamHandler {
    return &ssews.BaseStreamHandler{
        Writer:   writer,
        TraceID:  traceID,
        TenantID: tenantID,
    }
}
```

#### `backend/internal/handler/chat.go`
```go
package handler

import (
    "net/http"
    "github.com/gin-gonic/gin"
    "voicehelper/internal/ssews"
)

type ChatHandler struct {
    BaseHandler
}

func (h *ChatHandler) StreamChat(c *gin.Context) {
    // 创建 SSE 写入器
    writer := ssews.NewSSEWriter(c.Writer)
    if writer == nil {
        c.JSON(http.StatusInternalServerError, gin.H{"error": "SSE not supported"})
        return
    }
    defer writer.Close()
    
    // 提取追踪信息
    traceID, tenantID := h.extractTraceInfo(c)
    streamHandler := h.createStreamHandler(writer, traceID, tenantID)
    
    // 解析请求
    var req ChatRequest
    if err := c.ShouldBindJSON(&req); err != nil {
        streamHandler.Writer.WriteError("invalid_request", err.Error())
        return
    }
    
    // 转发到算法服务
    h.forwardToAlgoService(streamHandler, req)
}

func (h *ChatHandler) forwardToAlgoService(handler *ssews.BaseStreamHandler, req ChatRequest) {
    // TODO: 实现转发逻辑
    // 1. 构建到算法服务的请求
    // 2. 建立 SSE 连接
    // 3. 转发事件流
}

type ChatRequest struct {
    Query     string                 `json:"query"`
    SessionID string                 `json:"session_id"`
    Context   map[string]interface{} `json:"context,omitempty"`
}
```

### 3.3 功能清单（网关侧）

- [ ] **G-1 SSE 父类封装**
  - **任务**：实现 `StreamWriter` 接口和 `SSEWriter` 实现
  - **DoD**：SSE 断线重连，首 token <800ms
  - **文件**：`backend/internal/ssews/` 目录

- [ ] **G-2 WS 父类封装**
  - **任务**：实现 WebSocket 的 `StreamWriter` 实现
  - **DoD**：语音回放脚本通过，barge-in ≤150ms
  - **文件**：`backend/internal/ssews/ws.go`

- [ ] **G-3 Handler 子类**
  - **任务**：实现 `ChatHandler`, `VoiceHandler`, `AgentHandler`
  - **DoD**：OpenAPI 契约测试 & WS/SSE 回放通过
  - **文件**：`backend/internal/handler/` 目录

- [ ] **G-4 中间件**
  - **任务**：实现 Auth/RateLimit/Tracing 中间件
  - **DoD**：Prometheus 指标可见；429 行为正确
  - **文件**：`backend/internal/middleware/` 目录

**运行命令（网关）**：
```bash
cd backend && go run ./cmd/gateway
```

---

## 4. 前端（Node.js/Next.js）- 父类客户端设计

### 4.1 TS SDK 父类抽象

#### `frontend/src/api/base.ts`
```typescript
export interface EventSink {
  on(event: string, data: any): void;
}

export interface StreamOptions {
  timeout?: number;
  retries?: number;
  headers?: Record<string, string>;
}

export abstract class BaseStreamClient {
  protected headers: Record<string, string> = {};
  protected baseURL: string;
  
  constructor(baseURL: string = '', token?: string) {
    this.baseURL = baseURL;
    if (token) {
      this.headers.Authorization = `Bearer ${token}`;
    }
  }
  
  protected generateTraceId(): string {
    return `trace_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  protected addTraceHeaders(headers: Record<string, string> = {}): Record<string, string> {
    return {
      ...this.headers,
      'X-Trace-ID': this.generateTraceId(),
      'X-Tenant-ID': 'default',
      ...headers,
    };
  }
}
```

#### `frontend/src/api/chat.ts`
```typescript
import { BaseStreamClient, EventSink, StreamOptions } from './base';

export interface ChatRequest {
  query: string;
  session_id: string;
  context?: Record<string, any>;
}

export class ChatSSEClient extends BaseStreamClient {
  async streamChat(request: ChatRequest, sink: EventSink, options: StreamOptions = {}) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), options.timeout || 30000);
    
    try {
      const response = await fetch(`${this.baseURL}/api/v1/chat/stream`, {
        method: 'POST',
        headers: {
          ...this.addTraceHeaders(),
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
        signal: controller.signal,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            const event = line.substring(7);
            continue;
          }
          
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.substring(6));
              sink.on(event || 'message', data);
            } catch (e) {
              console.warn('Failed to parse SSE data:', line);
            }
          }
        }
      }
    } finally {
      clearTimeout(timeoutId);
    }
  }
  
  async cancelChat(sessionId: string) {
    await fetch(`${this.baseURL}/api/v1/chat/cancel`, {
      method: 'POST',
      headers: this.addTraceHeaders(),
      body: JSON.stringify({ session_id: sessionId }),
    });
  }
}
```

### 4.2 音频处理父类

#### `frontend/src/audio/base.ts`
```typescript
export interface AudioConfig {
  sampleRate: number;
  channels: number;
  bitDepth: number;
}

export abstract class BaseAudioProcessor {
  protected config: AudioConfig;
  protected isRecording: boolean = false;
  
  constructor(config: AudioConfig = { sampleRate: 16000, channels: 1, bitDepth: 16 }) {
    this.config = config;
  }
  
  abstract start(): Promise<void>;
  abstract stop(): Promise<void>;
  abstract onAudioData(callback: (data: ArrayBuffer) => void): void;
  
  protected validateConfig(config: AudioConfig): boolean {
    return config.sampleRate > 0 && config.channels > 0 && config.bitDepth > 0;
  }
}

export abstract class BasePlayer {
  protected isPlaying: boolean = false;
  protected queue: ArrayBuffer[] = [];
  
  abstract play(audioData: ArrayBuffer): Promise<void>;
  abstract stop(): void;
  abstract setVolume(volume: number): void;
  
  protected enqueue(data: ArrayBuffer) {
    this.queue.push(data);
  }
  
  protected dequeue(): ArrayBuffer | undefined {
    return this.queue.shift();
  }
}
```

#### `frontend/src/audio/pcm16.ts`
```typescript
import { BaseAudioProcessor, BasePlayer, AudioConfig } from './base';

export class PCM16Processor extends BaseAudioProcessor {
  private mediaRecorder?: MediaRecorder;
  private audioContext?: AudioContext;
  private workletNode?: AudioWorkletNode;
  private onDataCallback?: (data: ArrayBuffer) => void;
  
  async start(): Promise<void> {
    if (this.isRecording) return;
    
    const stream = await navigator.mediaDevices.getUserMedia({ 
      audio: {
        sampleRate: this.config.sampleRate,
        channelCount: this.config.channels,
      }
    });
    
    this.audioContext = new AudioContext({ sampleRate: this.config.sampleRate });
    
    // 加载 AudioWorklet
    await this.audioContext.audioWorklet.addModule('/audio-processor.js');
    
    this.workletNode = new AudioWorkletNode(this.audioContext, 'pcm16-processor');
    this.workletNode.port.onmessage = (event) => {
      if (this.onDataCallback && event.data.audioData) {
        this.onDataCallback(event.data.audioData);
      }
    };
    
    const source = this.audioContext.createMediaStreamSource(stream);
    source.connect(this.workletNode);
    
    this.isRecording = true;
  }
  
  async stop(): Promise<void> {
    if (!this.isRecording) return;
    
    this.workletNode?.disconnect();
    await this.audioContext?.close();
    this.isRecording = false;
  }
  
  onAudioData(callback: (data: ArrayBuffer) => void): void {
    this.onDataCallback = callback;
  }
}

export class PCMChunkPlayer extends BasePlayer {
  private audioContext?: AudioContext;
  private gainNode?: GainNode;
  private currentSource?: AudioBufferSourceNode;
  
  constructor() {
    super();
    this.audioContext = new AudioContext({ sampleRate: 16000 });
    this.gainNode = this.audioContext.createGain();
    this.gainNode.connect(this.audioContext.destination);
  }
  
  async play(audioData: ArrayBuffer): Promise<void> {
    if (!this.audioContext || !this.gainNode) return;
    
    const audioBuffer = await this.audioContext.decodeAudioData(audioData);
    const source = this.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(this.gainNode);
    
    this.currentSource = source;
    source.start();
    this.isPlaying = true;
    
    source.onended = () => {
      this.isPlaying = false;
      this.playNext();
    };
  }
  
  stop(): void {
    this.currentSource?.stop();
    this.queue = [];
    this.isPlaying = false;
  }
  
  setVolume(volume: number): void {
    if (this.gainNode) {
      this.gainNode.gain.value = Math.max(0, Math.min(1, volume));
    }
  }
  
  private async playNext() {
    const nextData = this.dequeue();
    if (nextData) {
      await this.play(nextData);
    }
  }
}
```

### 4.3 功能清单（前端侧）

- [ ] **F-1 ChatSSEClient 子类**
  - **任务**：实现 SSE 客户端，支持断线重连
  - **DoD**：断线重连脚本通过；UI 正常渲染
  - **文件**：`frontend/src/api/chat.ts`

- [ ] **F-2 VoiceWSClient 子类**
  - **任务**：实现 WebSocket 语音客户端
  - **DoD**：端到端闭环，拼接间隙 P95≤120ms
  - **文件**：`frontend/src/api/voice.ts`

- [ ] **F-3 音频处理器子类**
  - **任务**：实现 `PCM16Processor` 和 `PCMChunkPlayer`
  - **DoD**：音频采集和播放正常，延迟 <200ms
  - **文件**：`frontend/src/audio/` 目录

- [ ] **F-4 UI 组件**
  - **任务**：实现 Chat/Voice 统一会话界面
  - **DoD**：Playwright E2E 通过（文本/语音）
  - **文件**：`frontend/components/` 目录

**运行命令（前端）**：
```bash
cd frontend && npm install && npm run dev
```

---

## 5. API 契约定义

### 5.1 RESTful API

```yaml
# docs/openapi.yaml
openapi: 3.0.3
info:
  title: VoiceHelper API
  version: 2.0.0

paths:
  /api/v1/chat/stream:
    post:
      summary: 流式聊天
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                session_id:
                  type: string
                context:
                  type: object
      responses:
        '200':
          description: SSE 流
          content:
            text/event-stream:
              schema:
                type: string

  /api/v1/chat/cancel:
    post:
      summary: 取消聊天
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                session_id:
                  type: string

  /api/v1/voice/stream:
    get:
      summary: WebSocket 语音流
      parameters:
        - name: Upgrade
          in: header
          required: true
          schema:
            type: string
            enum: [websocket]
```

### 5.2 WebSocket 协议

```json
// 客户端 -> 服务端
{
  "type": "start",
  "data": {
    "session_id": "session_123",
    "config": {
      "sample_rate": 16000,
      "channels": 1,
      "language": "zh-CN"
    }
  }
}

{
  "type": "audio",
  "data": "<base64_encoded_pcm_data>"
}

{
  "type": "stop",
  "data": {
    "session_id": "session_123"
  }
}

// 服务端 -> 客户端
{
  "event": "asr_partial",
  "data": {
    "text": "你好",
    "confidence": 0.8
  }
}

{
  "event": "asr_final",
  "data": {
    "text": "你好世界",
    "confidence": 0.95
  }
}

{
  "event": "agent_response",
  "data": {
    "text": "你好！有什么可以帮助你的吗？",
    "references": []
  }
}

{
  "event": "tts_audio",
  "data": "<base64_encoded_audio_data>"
}
```

---

## 6. 测试与验收

### 6.1 单元测试

#### Python 测试
```bash
# 算法服务测试
cd algo
pytest tests/ -v --cov=core --cov-report=html
```

#### Go 测试
```bash
# 网关测试
cd backend
go test ./... -v -race -coverprofile=coverage.out
go tool cover -html=coverage.out
```

#### TypeScript 测试
```bash
# 前端测试
cd frontend
npm run test -- --coverage
```

### 6.2 集成测试

```bash
# 启动依赖服务
docker-compose -f deploy/docker-compose.local.yml up -d milvus redis

# 运行集成测试
./scripts/run_integration_tests.sh
```

### 6.3 E2E 测试

```bash
# Playwright E2E 测试
cd frontend
npx playwright test tests/e2e/
```

### 6.4 性能测试

```bash
# k6 压力测试
k6 run tests/performance/chat_load_test.js
k6 run tests/performance/voice_load_test.js
```

### 6.5 验收标准（DoD）

**性能指标**：
- 文本首 token < 800ms
- 语音首响 < 700ms  
- barge-in ≤ 150ms
- RAG 检索 P95 < 120ms
- TTS 首音 < 500ms

**质量指标**：
- 单测覆盖率 ≥ 70%
- 集成测试通过率 100%
- E2E 测试通过率 ≥ 95%
- 错误率 < 1%

**可观测性**：
- 所有接口有 Prometheus 指标
- 分布式追踪覆盖关键路径
- 错误日志结构化且可搜索

---

## 7. 扩展点设计

### 7.1 检索策略扩展
```python
class HybridRetriever(BaseRetriever):
    """混合检索器：向量 + BM25"""
    
class MultiVectorRetriever(BaseRetriever):
    """多向量检索器：支持多种嵌入模型"""
```

### 7.2 工具扩展
```python
class DatabaseTool(BaseTool):
    """数据库查询工具"""
    
class EmailTool(BaseTool):
    """邮件发送工具"""
```

### 7.3 ASR/TTS 供应商扩展
```python
class BaiduAsrAdapter(AsrStream):
    """百度 ASR 适配器"""
    
class XunfeiTtsAdapter(TtsStream):
    """讯飞 TTS 适配器"""
```

### 7.4 Agent 图扩展
```python
class ConfirmationAgentGraph(BaseAgentGraph):
    """带确认的 Agent 图"""
    
class MultiModalAgentGraph(BaseAgentGraph):
    """多模态 Agent 图"""
```

---

## 8. 部署与运维

### 8.1 Docker 镜像构建
```bash
# 构建所有服务镜像
make build-all

# 单独构建
make build-frontend
make build-backend  
make build-algo
```

### 8.2 本地开发环境
```bash
# 启动开发环境
make dev-up

# 停止开发环境
make dev-down
```

### 8.3 生产部署
```bash
# Kubernetes 部署
kubectl apply -f deploy/k8s/

# Helm 部署
helm install voicehelper deploy/helm/voicehelper
```

---

## 9. 执行时间线（两周 MVP）

### 第一周
- **Day 1-2**: 父类/协议/Mixin 实现
- **Day 3-4**: SSE/WS 管道和前端 SDK
- **Day 5**: RAG MilvusRetriever 子类

### 第二周  
- **Day 6-7**: 工具三件套和 LangGraph 子类图
- **Day 8-9**: ASR/TTS 适配器子类
- **Day 10**: UI 联调和 E2E 测试

---

## 10. 监控与观测

### 10.1 关键指标
- **业务指标**: 会话成功率、用户满意度、功能使用率
- **技术指标**: 响应时间、错误率、吞吐量、资源使用率
- **成本指标**: Token 消耗、API 调用次数、存储使用量

### 10.2 告警规则
- P95 响应时间 > 阈值
- 错误率 > 1%
- 服务可用性 < 99.9%
- Token 消耗异常增长

### 10.3 日志规范
```json
{
  "timestamp": "2025-09-22T10:00:00Z",
  "level": "INFO",
  "service": "algo",
  "trace_id": "trace_123",
  "message": "Tool execution completed",
  "duration_ms": 150,
  "tool_name": "fetch_url",
  "success": true
}
```

---

## 总结

本重构计划采用父类/子类设计模式，通过抽象基类沉淀公共逻辑，子类专注业务差异，Mixin 解耦横切能力。整个架构具备：

1. **高扩展性**: 新增工具、检索器、适配器只需继承对应父类
2. **强一致性**: 统一的接口契约和事件模型
3. **易测试性**: 父类逻辑可独立测试，子类只测差异部分
4. **可观测性**: 内置事件回调和性能监控
5. **高性能**: 流式处理、缓存、重试等优化

通过这种设计，系统既保持了架构的清晰性，又具备了良好的扩展能力，为后续功能迭代奠定了坚实基础。

<function_calls>
<invoke name="todo_write">
<parameter name="merge">true
