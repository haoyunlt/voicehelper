"""
BGE 向量化服务
提供基于BGE模型的文本向量化功能
"""

import os
import time
import asyncio
from typing import List, Optional
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus 指标
REQUEST_COUNT = Counter('bge_requests_total', 'Total BGE requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('bge_request_duration_seconds', 'BGE request duration')
EMBEDDING_COUNT = Counter('bge_embeddings_total', 'Total embeddings generated')

# 全局变量
model = None

class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="要向量化的文本列表")
    normalize: bool = Field(True, description="是否归一化向量")
    batch_size: Optional[int] = Field(None, description="批处理大小")

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="向量列表")
    dimension: int = Field(..., description="向量维度")
    model_name: str = Field(..., description="模型名称")
    processing_time: float = Field(..., description="处理时间(秒)")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    device: str
    memory_usage: Optional[dict] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时加载模型
    global model
    model_name = os.getenv("MODEL_NAME", "BAAI/bge-large-zh-v1.5")
    cache_dir = os.getenv("MODEL_CACHE_DIR", "/app/models")
    
    logger.info(f"Loading BGE model: {model_name}")
    try:
        model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Model loaded successfully on {model.device}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # 关闭时清理
    if model:
        del model

app = FastAPI(
    title="BGE Vector Embedding Service",
    description="BGE向量化服务，支持中英文文本向量化",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    memory_info = None
    if torch.cuda.is_available():
        memory_info = {
            "allocated": torch.cuda.memory_allocated(),
            "cached": torch.cuda.memory_reserved(),
            "max_allocated": torch.cuda.max_memory_allocated()
        }
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_name=os.getenv("MODEL_NAME", "BAAI/bge-large-zh-v1.5"),
        device=str(model.device) if model else "unknown",
        memory_usage=memory_info
    )

@app.get("/ready")
async def ready_check():
    """就绪检查接口"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

@app.get("/metrics")
async def metrics():
    """Prometheus指标接口"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/embed", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """创建文本向量"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    REQUEST_COUNT.labels(method="POST", endpoint="/embed").inc()
    
    start_time = time.time()
    
    try:
        # 参数验证
        if not request.texts:
            raise HTTPException(status_code=400, detail="texts cannot be empty")
        
        if len(request.texts) > 100:
            raise HTTPException(status_code=400, detail="Too many texts (max 100)")
        
        # 生成向量
        batch_size = request.batch_size or int(os.getenv("MAX_BATCH_SIZE", "32"))
        
        embeddings = model.encode(
            request.texts,
            batch_size=batch_size,
            normalize_embeddings=request.normalize,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # 转换为列表格式
        embeddings_list = embeddings.tolist()
        
        processing_time = time.time() - start_time
        
        EMBEDDING_COUNT.inc(len(request.texts))
        REQUEST_DURATION.observe(processing_time)
        
        return EmbeddingResponse(
            embeddings=embeddings_list,
            dimension=len(embeddings_list[0]) if embeddings_list else 0,
            model_name=os.getenv("MODEL_NAME", "BAAI/bge-large-zh-v1.5"),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def get_info():
    """获取服务信息"""
    return {
        "service": "BGE Vector Embedding Service",
        "model_name": os.getenv("MODEL_NAME", "BAAI/bge-large-zh-v1.5"),
        "max_batch_size": int(os.getenv("MAX_BATCH_SIZE", "32")),
        "max_sequence_length": int(os.getenv("MAX_SEQUENCE_LENGTH", "512")),
        "device": str(model.device) if model else "unknown",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
