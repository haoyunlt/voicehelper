"""
FAISS 向量搜索服务
提供基于FAISS的高效向量相似度搜索功能
"""

import os
import json
import time
import asyncio
import threading
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus 指标
REQUEST_COUNT = Counter('faiss_requests_total', 'Total FAISS requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('faiss_request_duration_seconds', 'FAISS request duration')
INDEX_SIZE = Gauge('faiss_index_size', 'Number of vectors in index')
SEARCH_COUNT = Counter('faiss_searches_total', 'Total searches performed')

# 全局变量
index = None
vector_ids = []
metadata_store = {}
index_lock = threading.RLock()

class AddVectorsRequest(BaseModel):
    vectors: List[List[float]] = Field(..., description="向量列表")
    ids: Optional[List[str]] = Field(None, description="向量ID列表")
    metadata: Optional[List[Dict[str, Any]]] = Field(None, description="元数据列表")

class SearchRequest(BaseModel):
    query_vector: List[float] = Field(..., description="查询向量")
    k: int = Field(10, description="返回结果数量", ge=1, le=100)
    nprobe: Optional[int] = Field(None, description="搜索探针数量")

class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_time: float
    total_vectors: int

class IndexStats(BaseModel):
    total_vectors: int
    dimension: int
    index_type: str
    is_trained: bool
    memory_usage_mb: float

def create_faiss_index(dimension: int, index_type: str = "IVF") -> faiss.Index:
    """创建FAISS索引"""
    nlist = int(os.getenv("FAISS_NLIST", "100"))
    
    if index_type == "IVF":
        # IVF索引，适合大规模数据
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    elif index_type == "HNSW":
        # HNSW索引，适合高精度搜索
        index = faiss.IndexHNSWFlat(dimension, 32)
    else:
        # 默认使用Flat索引
        index = faiss.IndexFlatL2(dimension)
    
    return index

def save_index():
    """保存索引到磁盘"""
    try:
        with index_lock:
            if index is not None and index.ntotal > 0:
                faiss.write_index(index, "/app/data/faiss.index")
                
                # 保存元数据
                metadata = {
                    "vector_ids": vector_ids,
                    "metadata_store": metadata_store,
                    "dimension": index.d,
                    "total_vectors": index.ntotal
                }
                
                with open("/app/data/metadata.json", "w") as f:
                    json.dump(metadata, f)
                
                logger.info(f"Index saved with {index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Failed to save index: {e}")

def load_index():
    """从磁盘加载索引"""
    global index, vector_ids, metadata_store
    
    try:
        if os.path.exists("/app/data/faiss.index"):
            index = faiss.read_index("/app/data/faiss.index")
            logger.info(f"Loaded index with {index.ntotal} vectors")
            
            # 加载元数据
            if os.path.exists("/app/data/metadata.json"):
                with open("/app/data/metadata.json", "r") as f:
                    metadata = json.load(f)
                    vector_ids = metadata.get("vector_ids", [])
                    metadata_store = metadata.get("metadata_store", {})
            
            INDEX_SIZE.set(index.ntotal)
            return True
    except Exception as e:
        logger.error(f"Failed to load index: {e}")
    
    return False

async def periodic_save():
    """定期保存索引"""
    interval = int(os.getenv("INDEX_SAVE_INTERVAL", "300"))
    while True:
        await asyncio.sleep(interval)
        save_index()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global index
    
    # 启动时初始化
    os.makedirs("/app/data", exist_ok=True)
    
    # 尝试加载现有索引
    if not load_index():
        # 创建新索引
        dimension = int(os.getenv("VECTOR_DIMENSION", "1024"))
        index_type = os.getenv("FAISS_INDEX_TYPE", "IVF")
        index = create_faiss_index(dimension, index_type)
        logger.info(f"Created new {index_type} index with dimension {dimension}")
    
    # 启动定期保存任务
    save_task = asyncio.create_task(periodic_save())
    
    yield
    
    # 关闭时保存索引
    save_task.cancel()
    save_index()

app = FastAPI(
    title="FAISS Vector Search Service",
    description="FAISS向量搜索服务，支持高效的相似度搜索",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "index_loaded": index is not None,
        "total_vectors": index.ntotal if index else 0
    }

@app.get("/ready")
async def ready_check():
    """就绪检查接口"""
    if index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    return {"status": "ready"}

@app.get("/metrics")
async def metrics():
    """Prometheus指标接口"""
    if index:
        INDEX_SIZE.set(index.ntotal)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/add")
async def add_vectors(request: AddVectorsRequest):
    """添加向量到索引"""
    if index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    REQUEST_COUNT.labels(method="POST", endpoint="/add").inc()
    start_time = time.time()
    
    try:
        vectors = np.array(request.vectors, dtype=np.float32)
        
        if vectors.shape[1] != index.d:
            raise HTTPException(
                status_code=400,
                detail=f"Vector dimension {vectors.shape[1]} doesn't match index dimension {index.d}"
            )
        
        with index_lock:
            # 训练索引（如果需要）
            if hasattr(index, 'is_trained') and not index.is_trained:
                if vectors.shape[0] >= index.nlist:
                    index.train(vectors)
                    logger.info("Index trained successfully")
                else:
                    logger.warning(f"Not enough vectors to train index (need {index.nlist}, got {vectors.shape[0]})")
            
            # 添加向量
            start_id = len(vector_ids)
            index.add(vectors)
            
            # 更新ID和元数据
            if request.ids:
                if len(request.ids) != len(request.vectors):
                    raise HTTPException(status_code=400, detail="IDs count doesn't match vectors count")
                vector_ids.extend(request.ids)
            else:
                vector_ids.extend([f"vec_{start_id + i}" for i in range(len(request.vectors))])
            
            if request.metadata:
                if len(request.metadata) != len(request.vectors):
                    raise HTTPException(status_code=400, detail="Metadata count doesn't match vectors count")
                for i, meta in enumerate(request.metadata):
                    metadata_store[vector_ids[start_id + i]] = meta
        
        processing_time = time.time() - start_time
        REQUEST_DURATION.observe(processing_time)
        
        return {
            "message": f"Added {len(request.vectors)} vectors",
            "total_vectors": index.ntotal,
            "processing_time": processing_time
        }
        
    except Exception as e:
        logger.error(f"Error adding vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=SearchResponse)
async def search_vectors(request: SearchRequest):
    """搜索相似向量"""
    if index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    if index.ntotal == 0:
        return SearchResponse(results=[], query_time=0.0, total_vectors=0)
    
    REQUEST_COUNT.labels(method="POST", endpoint="/search").inc()
    SEARCH_COUNT.inc()
    start_time = time.time()
    
    try:
        query_vector = np.array([request.query_vector], dtype=np.float32)
        
        if query_vector.shape[1] != index.d:
            raise HTTPException(
                status_code=400,
                detail=f"Query vector dimension {query_vector.shape[1]} doesn't match index dimension {index.d}"
            )
        
        # 设置搜索参数
        if hasattr(index, 'nprobe') and request.nprobe:
            index.nprobe = request.nprobe
        
        with index_lock:
            # 执行搜索
            scores, indices = index.search(query_vector, min(request.k, index.ntotal))
        
        # 构建结果
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx >= 0 and idx < len(vector_ids):  # 有效索引
                vector_id = vector_ids[idx]
                metadata = metadata_store.get(vector_id)
                results.append(SearchResult(
                    id=vector_id,
                    score=float(score),
                    metadata=metadata
                ))
        
        query_time = time.time() - start_time
        REQUEST_DURATION.observe(query_time)
        
        return SearchResponse(
            results=results,
            query_time=query_time,
            total_vectors=index.ntotal
        )
        
    except Exception as e:
        logger.error(f"Error searching vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=IndexStats)
async def get_stats():
    """获取索引统计信息"""
    if index is None:
        raise HTTPException(status_code=503, detail="Index not initialized")
    
    # 估算内存使用
    memory_usage = 0
    if hasattr(index, 'ntotal') and index.ntotal > 0:
        memory_usage = index.ntotal * index.d * 4 / (1024 * 1024)  # 假设float32
    
    return IndexStats(
        total_vectors=index.ntotal,
        dimension=index.d,
        index_type=type(index).__name__,
        is_trained=getattr(index, 'is_trained', True),
        memory_usage_mb=memory_usage
    )

@app.delete("/clear")
async def clear_index():
    """清空索引"""
    global index, vector_ids, metadata_store
    
    with index_lock:
        if index:
            dimension = index.d
            index_type = os.getenv("FAISS_INDEX_TYPE", "IVF")
            index = create_faiss_index(dimension, index_type)
            vector_ids = []
            metadata_store = {}
    
    return {"message": "Index cleared successfully"}

@app.post("/save")
async def save_index_manually():
    """手动保存索引"""
    save_index()
    return {"message": "Index saved successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
