# BGE + FAISS RAG 使用指南

## 概述

本系统基于 **BGE (BAAI General Embedding)** 嵌入模型和 **FAISS (Facebook AI Similarity Search)** 向量数据库，提供高效的中文检索增强生成（RAG）功能。

## 核心特性

- **中文优化**: 使用 BAAI/bge-large-zh-v1.5 模型，专为中文语义理解优化
- **高性能检索**: FAISS HNSW 索引，P95 延迟 < 120ms
- **租户隔离**: 支持多租户数据隔离
- **指标监控**: 完整的性能指标收集和监控
- **灵活配置**: 支持多种索引类型和参数调优

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基础使用

```python
from core.rag_factory import get_rag_factory

# 获取RAG工厂
factory = get_rag_factory()

# 准备文档
documents = [
    {
        "id": "doc_1",
        "title": "文档标题",
        "content": "文档内容...",
        "source": "来源",
        "metadata": {"category": "技术"}
    }
]

# 构建索引
result = await factory.build_index_from_documents(
    documents=documents,
    tenant_id="my_tenant",
    dataset_id="my_dataset"
)

# 创建检索器
retriever = factory.create_retriever(
    tenant_id="my_tenant",
    dataset_id="my_dataset"
)

# 执行检索
results = retriever.retrieve(
    query="你的问题",
    top_k=5
)
```

### 3. 集成到Agent

```python
from core.langgraph_agent import LangGraphAgent
from core.rag_factory import create_retriever

# 创建检索器
retriever = create_retriever(
    tenant_id="agent_tenant",
    dataset_id="knowledge_base"
)

# 创建Agent
agent = LangGraphAgent(
    llm_service=llm_service,
    retriever=retriever,  # 使用新的检索器
    tools=tools
)
```

## 配置说明

### 环境变量配置

```bash
# BGE模型配置
BGE_MODEL_NAME=BAAI/bge-large-zh-v1.5
BGE_DEVICE=cpu
BGE_NORMALIZE=true
BGE_BATCH_SIZE=32

# FAISS配置
FAISS_INDEX_TYPE=HNSW32,Flat
FAISS_EF_CONSTRUCTION=200
FAISS_EF_SEARCH=64
FAISS_DATA_DIR=data/faiss
FAISS_TENANT_BASED=true

# 文档处理配置
DOC_CHUNK_SIZE=512
DOC_CHUNK_OVERLAP=50

# RAG系统配置
RAG_ENABLE_METRICS=true
RAG_ENABLE_CACHE=true
```

### 代码配置

```python
from core.config.bge_config import RAGConfig, BGEConfig, FAISSConfig

config = RAGConfig(
    bge=BGEConfig(
        model_name="BAAI/bge-large-zh-v1.5",
        device="cuda",
        normalize=True
    ),
    faiss=FAISSConfig(
        index_type="HNSW32,Flat",
        ef_construction=200,
        ef_search=64
    )
)
```

## API 使用

### 文档入库

```python
from core.api_endpoints import rag_endpoints

# 入库请求
request = IngestRequest(
    documents=[
        DocumentInput(
            id="doc_1",
            title="标题",
            content="内容",
            source="来源"
        )
    ],
    tenant_id="my_tenant",
    dataset_id="my_dataset"
)

result = await rag_endpoints.ingest_documents(request)
```

### 文档检索

```python
# 检索请求
request = RetrieveRequest(
    query="你的问题",
    tenant_id="my_tenant",
    dataset_id="my_dataset",
    top_k=5
)

result = await rag_endpoints.retrieve_documents(request)
```

### 指标查询

```python
# 检索指标
retrieval_metrics = rag_endpoints.get_retrieval_metrics(window_minutes=5)

# 索引指标
index_metrics = rag_endpoints.get_index_metrics()

# 缓存指标
cache_metrics = rag_endpoints.get_cache_metrics()
```

## 性能优化

### 1. 模型选择

- **bge-large-zh-v1.5**: 最佳质量，1024维，适合生产环境
- **bge-base-zh-v1.5**: 平衡性能，768维，适合中等规模
- **bge-small-zh-v1.5**: 快速推理，512维，适合资源受限环境

### 2. 索引类型

- **HNSW32,Flat**: 高质量检索，适合大规模数据
- **Flat**: 精确搜索，适合小规模数据
- **IVF1024,Flat**: 倒排索引，适合超大规模数据

### 3. 参数调优

```python
# 构建时参数
ef_construction = 200  # 越大质量越好，构建越慢
batch_size = 32       # 根据GPU内存调整

# 检索时参数
ef_search = 64        # 越大质量越好，检索越慢
top_k = 5            # 返回结果数量
score_threshold = 0.3 # 分数阈值过滤
```

## 监控指标

### 检索指标

- **平均延迟**: avg_latency_ms
- **P95延迟**: p95_latency_ms  
- **平均结果数**: avg_results_count
- **平均分数**: avg_score
- **缓存命中率**: cache_hit_rate

### 索引指标

- **加载时间**: load_time_ms
- **索引大小**: index_size_mb
- **向量数量**: vector_count
- **维度**: dimension

### 质量目标

- **召回率**: Recall@5 ≥ 0.85
- **检索延迟**: P95 < 120ms
- **入库性能**: < 10分钟/10万段
- **内存控制**: < 3GB/10万段

## 故障排除

### 常见问题

1. **模型加载失败**
   ```
   解决方案: 检查网络连接，确保有足够磁盘空间
   ```

2. **CUDA内存不足**
   ```
   解决方案: 减小batch_size，或使用CPU模式
   ```

3. **检索结果质量差**
   ```
   解决方案: 调整chunk_size，增加ef_search参数
   ```

4. **索引加载慢**
   ```
   解决方案: 使用SSD存储，预热索引
   ```

### 日志分析

```python
from loguru import logger

# 启用详细日志
logger.add("rag.log", level="DEBUG")

# 关键日志点
- "BGE模型加载完成": 模型初始化成功
- "索引已加载": 索引加载成功  
- "检索完成": 检索执行成功
- "记录检索指标": 指标收集正常
```

## 最佳实践

### 1. 文档预处理

```python
# 清理文本
def clean_text(text: str) -> str:
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    return text.strip()

# 优化分块
chunker = DocumentChunker(
    chunk_size=512,      # 适中的块大小
    chunk_overlap=50,    # 保持上下文连续性
    separators=['。', '！', '？', '\n\n']  # 语义边界分割
)
```

### 2. 索引管理

```python
# 定期重建索引
async def rebuild_index_if_needed():
    stats = factory.get_retriever_stats()
    if stats.get("total_vectors", 0) > 100000:
        # 重建索引以优化性能
        await factory.build_index_from_documents(documents)

# 索引版本管理
def save_index_with_version(dataset_id: str, version: str):
    versioned_id = f"{dataset_id}_v{version}"
    # 保存到版本化路径
```

### 3. 查询优化

```python
# 查询扩展
def expand_query(query: str) -> List[str]:
    # 添加同义词
    # 生成相关问题
    return [query] + expanded_queries

# 结果后处理
def post_process_results(results: List[Dict]) -> List[Dict]:
    # 去重
    # 重排序
    # 过滤低质量结果
    return filtered_results
```

## 示例代码

完整示例请参考:
- `algo/examples/bge_faiss_example.py` - 基础使用示例
- `algo/tests/test_bge_faiss_integration.py` - 集成测试示例

## 版本历史

- **v2.1.0**: 初始BGE+FAISS实现
- **v2.1.1**: 添加指标监控
- **v2.1.2**: 优化性能和稳定性
