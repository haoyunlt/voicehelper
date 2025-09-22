# BGE Embedding 迁移计划

## 变更摘要

从 **Milvus 向量库** 替换为 **BGE Embedding + FAISS**

- **检索底座**：BGE Embedding（HuggingFace/SentenceTransformers）+ FAISS
- **算法侧**：新增 `BgeEmbedder` 与 `FaissRetriever` 子类
- **数据侧**：索引文件落地到对象存储/本地卷，支持租户分片
- **API/协议**：对外接口不变，仅实现细节更换
- **质量目标**：中文优先模型，Recall@5≥0.85，检索 P95 < 120ms

## 技术实现

### 1. 新增 BgeEmbedder 类

```python
# algo/core/rag/embedder_bge.py
from sentence_transformers import SentenceTransformer
import numpy as np

class BgeEmbedder:
    model_name: str = "BAAI/bge-large-zh-v1.5"
    device: str = "cuda"
    normalize: bool = True
    
    def embed_queries(self, queries: list[str]) -> np.ndarray:
        qs = ["为这个问题生成表示用于检索相关文档：" + q for q in queries]
        return self.model().encode(qs, normalize_embeddings=self.normalize)
    
    def embed_passages(self, passages: list[str]) -> np.ndarray:
        ps = ["为这段文本生成表示用于检索相关文档：" + p for p in passages]
        return self.model().encode(ps, normalize_embeddings=self.normalize)
```

### 2. 新增 FaissRetriever 类

```python
# algo/core/rag/retriever_faiss.py
import faiss, json
from .base import BaseRetriever

class FaissRetriever(BaseRetriever):
    def retrieve(self, query: str, **kwargs) -> list[dict]:
        idx, meta = self._load()
        q = self.embedder.embed_queries([query])
        D, I = idx.search(q.astype("float32"), top_k)
        return [{"content": meta[i]["text"], "score": float(score)} 
                for i, score in zip(I[0], D[0])]
```

### 3. 索引构建

```python
# algo/core/rag/ingest_faiss.py
def build_faiss_index(passages: list[dict], embedder, index_out: str, meta_out: str):
    texts = [p["text"] for p in passages]
    X = embedder.embed_passages(texts).astype("float32")
    index = faiss.index_factory(X.shape[1], "HNSW32,Flat")
    index.add(X)
    faiss.write_index(index, index_out)
    with open(meta_out, "w") as f:
        json.dump(passages, f, ensure_ascii=False)
```

## 数据存储

```
tenants/{tenant_id}/datasets/{dataset_id}/
├── index.faiss    # FAISS 索引文件
└── meta.json      # 元数据文件
```

## 配置参数

```yaml
bge:
  model_name: "BAAI/bge-large-zh-v1.5"
  device: "cuda"
  normalize: true

faiss:
  index_type: "HNSW32,Flat"
  ef_construction: 200
  ef_search: 64
```

## 质量目标

- **召回率**：Recall@5 ≥ 0.85
- **检索延迟**：P95 < 120ms（10万段）
- **入库性能**：< 10分钟/10万段（GPU）
- **内存控制**：< 3GB/10万段

## 功能清单

### 算法侧
- [ ] A-1 引入 BGE：`BgeEmbedder` 类与依赖
- [ ] A-2 FAISS 检索器：`FaissRetriever` 实现
- [ ] A-3 Ingest 管线：`build_faiss_index()` + 切块器
- [ ] A-4 Agent 图切换：`RetrieveNode` 使用 `FaissRetriever`
- [ ] A-5 指标监控：延迟、加载时间等指标

### 网关侧
- [ ] G-1 /search 转发：查询参数透传
- [ ] G-2 /ingest 任务：记录索引路径
- [ ] G-3 指标监控：代理延迟指标

### 前端侧
- [ ] F-1 索引状态：展示索引信息
- [ ] F-2 检索预览：BGE 标识

## API 兼容性

### /api/v1/search
```json
{
  "records": [
    {"content": "...", "score": 0.83, "source": "doc_1#p12", "id": "chunk_001"}
  ]
}
```

### /api/v1/ingest/tasks/:id
```json
{
  "task_id": "t_xxx",
  "status": "succeeded",
  "index_path": "tenants/tn_x/datasets/ds_01/index.faiss",
  "meta_path": "tenants/tn_x/datasets/ds_01/meta.json",
  "dim": 1024,
  "doc_count": 100234
}
```

## 时间表

- **Week 1-2**: BGE Embedder + FAISS Retriever 实现
- **Week 3-4**: Agent 图集成 + 端到端测试
- **Week 5-6**: 性能优化 + 生产部署

## 风险与回退

**风险**：
- 内存占用较高 → 分片加载、PQ 压缩
- CPU 性能慢 → 异步入库、使用 bge-small-zh

**回退**：
- 保持 `VectorStorePort` 抽象
- 可切回 Milvus 或切换 pgvector

---

*版本: v2.1.0 | 创建: 2025-09-22*
